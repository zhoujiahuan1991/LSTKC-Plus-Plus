from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy

from reid.metric_learning.distance import cosine_similarity
class Trainer(object):
    def __init__(self,cfg,args, model, num_classes, writer=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        self.loss_ce=nn.CrossEntropyLoss(reduction='none')

        # self.proto_momentum =args.proto_momentum 

      
        # self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss( reduction = "none")
        self.MSE=torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None, 
              model_last=None, model_long=None):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_last = AverageMeter()
        losses_long = AverageMeter()

        end = time.time()
        
        for i in range(train_iters):
            train_inputs = data_loader_train.next()

            s_inputs, targets, cids, domains, = self._parse_data(train_inputs)
            
            targets =targets+ add_num
            s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)

            '''calculate the base loss'''
            loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
            
            loss_ce=self.loss_ce(cls_outputs, targets)

            loss=0
            weight=torch.ones_like(loss_ce)

            losses_ce.update(loss_ce.mean().item())
            losses_tr.update(loss_tp.item())

            af_loss=0
            af_items=0
            if model_last is not None:
                with torch.no_grad():
                    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = model_last(s_inputs, get_all_feat=True)
                    
                if isinstance(s_features_old, tuple):
                    s_features_old=s_features_old[0]
                
                Affinity_matrix_new = self.get_normal_affinity(s_features, self.args.tau)  #
                Affinity_matrix_old = self.get_normal_affinity(s_features_old, self.args.tau)
                Affinity_matrix_old_short=Affinity_matrix_old
                # print(Affinity_matrix_new[0].cpu().tolist())
                divergence, weight, Target_2 = self.cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old, targets)
                # loss = loss + divergence * self.AF_weight
                af_loss+=divergence
                af_items+=1
                losses_last.update(divergence.item())
            if model_long is not None:
                with torch.no_grad():
                    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = model_long(s_inputs, get_all_feat=True)
                if isinstance(s_features_old, tuple):
                    s_features_old=s_features_old[0]
                
                Affinity_matrix_new = self.get_normal_affinity(s_features, self.args.tau)  #
                Affinity_matrix_old = self.get_normal_affinity(s_features_old, self.args.tau)
                Affinity_matrix_old_long=Affinity_matrix_old
                # print(Affinity_matrix_new[0].cpu().tolist())
                divergence, weight, Target_1 = self.cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old, targets)
                # loss = loss + divergence * self.AF_weight
                af_loss+=divergence
                af_items+=1
                losses_long.update(divergence.item()) 
            model_fuse=True
            if af_items>1 and model_fuse:
                Target_1=(Target_1+Target_2)/2
                Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
                divergence=self.KLDivLoss(Affinity_matrix_new_log, Target_1)  # 128*128
                divergence=divergence.sum()/Affinity_matrix_new.size(0)
                af_loss=divergence*af_items
            comp_loss=True
            if comp_loss:
                if af_items>1:
                    divergence,_,_=self.dual_cal_KL_old_only(Affinity_matrix_new, Affinity_matrix_old_short,Affinity_matrix_old_long ,targets)
                    af_loss=divergence*af_items
            loss=loss+af_loss/(af_items+1e-6)*self.AF_weight


            loss = loss+ (loss_ce*weight).mean() + loss_tp
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()           

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                          global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                          global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                          global_step=epoch * train_iters + i)
            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tp {:.3f} ({:.3f})\t'
                      'Loss_last {:.3f} ({:.3f})\t'
                      'Loss_long {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_last.val, losses_last.avg,
                              losses_long.val, losses_long.avg,
                  ))       

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains
    def get_correctness(self,Affinity_matrix_new, Affinity_matrix_old,targets, Gts=None):
        if Gts == None:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        
        correct_old=(attri_old['FP'].sum(-1)*attri_old['FN'].sum(-1))==0

        

        return correct_old 
    def cal_KL_old_only(self,Affinity_matrix_new, Affinity_matrix_old,targets, Gts=None,):
        if Gts == None:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        # attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old = self.get_attri(Gts, Affinity_matrix_old, margin=0)

        '''# prediction is correct on old model'''
        Old_Keep = attri_old['TN'] + attri_old['TP']
        # if torch.any(Old_Keep<1):
        #     print(Old_Keep)
        Target_1 = Affinity_matrix_old * Old_Keep
        # '''# prediction is false on old model but correct on mew model'''
        # New_keep = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])
        # Target_2 = Affinity_matrix_new * New_keep
        '''# missed correct person'''
        Hard_pos = attri_old['FN']
        Thres_P = attri_old['Thres_P']
        Target_3 = Hard_pos * Thres_P

        '''# false wrong person'''
        Hard_neg = attri_old['FP']
        Thres_N = attri_old['Thres_N']
        Target_4 = Hard_neg * Thres_N

        Target__ = Target_1 +  Target_3 + Target_4
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization

       
      
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)  # 128*128

        # print("KL divergence",divergence.shape)
        
        if self.args.weighted_loss:
            Affinity_matrix_new=Affinity_matrix_new/(Affinity_matrix_new.max(-1, keepdim=True)[0]+1e-6)
            weight=torch.abs(Gts-Affinity_matrix_new)
            
            divergence=divergence*weight
            weight=weight.mean(-1)
        else:
            weight=torch.ones(divergence.shape[0]).to(divergence.device)
        divergence=divergence.sum()/Affinity_matrix_new.size(0)
        # divergence=divergence.mean()

        return divergence,weight, Target
    def dual_cal_KL_old_only(self,Affinity_matrix_new, Affinity_matrix_old_short,Affinity_matrix_old_long ,targets, Gts=None):
        if Gts == None:
            Gts = (targets.reshape(-1, 1) - targets.reshape(1, -1)) == 0  # Gt-matrix
            Gts = Gts.float().to(targets.device)
        '''obtain TP,FP,TN,FN'''
        # attri_new = self.get_attri(Gts, Affinity_matrix_new, margin=0)
        attri_old_short = self.get_attri(Gts, Affinity_matrix_old_short, margin=0)

        attri_old_long = self.get_attri(Gts, Affinity_matrix_old_long, margin=0)

        dual_keep=(attri_old_short['TN'] + attri_old_short['TP'])*(attri_old_long['TN'] + attri_old_long['TP'])

        Target_1=(Affinity_matrix_old_short*dual_keep+Affinity_matrix_old_long*dual_keep)/2

        single_keep_short=(attri_old_short['TN'] + attri_old_short['TP'])-(attri_old_long['TN'] + attri_old_long['TP'])   # 单个正确,取异或
        single_keep_short=single_keep_short.clamp(min=0.0)  # 取正数值
        Target_2=Affinity_matrix_old_short*single_keep_short

        single_keep_long=(attri_old_long['TN'] + attri_old_long['TP']) -(attri_old_short['TN'] + attri_old_short['TP'])  # 单个正确,取异或
        single_keep_long=single_keep_long.clamp(min=0.0)  # 取正数值
        Target_3=Affinity_matrix_old_long*single_keep_long


        '''# both missed correct person'''
        Hard_pos = attri_old_short['FN'] * attri_old_long['FN']
        Thres_P = torch.maximum(attri_old_short['Thres_P'], attri_old_long['Thres_P'])
        Target_4 = Hard_pos * Thres_P

        '''# both false wrong person'''
        Hard_neg = attri_old_short['FP'] * attri_old_long['FP']
        Thres_N = torch.minimum(attri_old_short['Thres_N'], attri_old_long['Thres_N'])
        Target_5 = Hard_neg * Thres_N


        Target__ = Target_1 +Target_2+  Target_3 + Target_4+Target_5
        Target = Target__ / (Target__.sum(1, keepdim=True))  # score normalization

       

        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)  # 128*128

        # print("KL divergence",divergence.shape)
        
        if self.args.weighted_loss:
            Affinity_matrix_new=Affinity_matrix_new/(Affinity_matrix_new.max(-1, keepdim=True)[0]+1e-6)
            weight=torch.abs(Gts-Affinity_matrix_new)
            
            divergence=divergence*weight
            weight=weight.mean(-1)
        else:
            weight=torch.ones(divergence.shape[0]).to(divergence.device)
        divergence=divergence.sum()/Affinity_matrix_new.size(0)
        # divergence=divergence.mean()
        if self.args.mse:
            divergence=self.MSE(Target, Affinity_matrix_new)*3000
        elif self.args.mae:
            divergence=self.MAE(Target,Affinity_matrix_new)
        elif self.args.js:
            Target_log=torch.log(Target)
            divergence1=self.KLDivLoss(Target_log, Affinity_matrix_new)  # 128*128
            divergence1=divergence1.sum()/Affinity_matrix_new.size(0)
            divergence=(divergence1+divergence)/2        
        else:
            pass


        return divergence,weight, Target
    

    def get_attri(self, Gts, pre_affinity_matrix,margin=0):
        Thres_P=((1-Gts)*pre_affinity_matrix).max(dim=1,keepdim=True)[0]
        T_scores=pre_affinity_matrix*Gts

        TP=((T_scores-Thres_P)>margin).float()
        try:
            TP=torch.maximum(TP, torch.eye(TP.size(0)).to(TP.device))
        except:
            pass

        FN=Gts-TP
        
        Mapped_affinity=(1-Gts) +pre_affinity_matrix
        try:
            Mapped_affinity = Mapped_affinity+torch.eye(Mapped_affinity.size(0)).to(Mapped_affinity.device)
        except:
            pass
        Thres_N = Mapped_affinity.min(dim=1, keepdim=True)[0]
        N_scores=pre_affinity_matrix*(1-Gts)

        FP=(N_scores>Thres_N ).float()
        TN=(1-Gts) -FP
        attris={
            'TP':TP,
            'FN':FN,
            'FP':FP,
            'TN':TN,
            "Thres_P":Thres_P,
            "Thres_N":Thres_N
        }
        return attris

