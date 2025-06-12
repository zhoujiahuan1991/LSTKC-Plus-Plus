from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

import torch.nn as nn
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s, fast_eval
import datetime
def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content
def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    # log_name = 'log.txt'
    timestamp = cur_timestamp_str()
    log_name = f'log_{timestamp}.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name=f'log_res_{timestamp}.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    elif 51 == args.setting:
        training_set = ['msmt17', 'cuhk_sysu', 'dukemtmc', 'market1501', 'cuhk03']
    elif 52 == args.setting:
        training_set = ['dukemtmc', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu']
    elif 53 == args.setting:
        training_set = ['cuhk_sysu', 'dukemtmc', 'cuhk03', 'msmt17', 'market1501']
    elif 54 == args.setting:
        training_set = ['cuhk03', 'msmt17', 'dukemtmc', 'market1501', 'cuhk_sysu']
    elif 55 == args.setting:
        training_set = ['market1501', 'msmt17', 'dukemtmc', 'cuhk_sysu', 'cuhk03']
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)

    model.cuda()
    model = DataParallel(model)    
    writer = SummaryWriter(log_dir=args.logs_dir)
    

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    elif args.MODEL in ['vit']:
        out_channel = 768
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")


    # train on the datasets squentially    
    # model: inference model
    # model_last: short-term old model
    # model_long: long-term old model
    # model_old: updated long-term old model
    model_last=None
    model_long=None
    model_old=None
    for set_index in range(0, len(training_set)):     
        # model_old = copy.deepcopy(model)
        if args.resume != '' and set_index==0:
            model_last=copy.deepcopy(model)
            model_new=model
            # continue
        elif args.resume_folder and set_index<=args.resume_id:
            ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
            checkpoint = load_checkpoint(args.resume_folder+'/'+ckpt_name[set_index])
            model.module.classifier = nn.Linear(out_channel, 500*(set_index+1), bias=False)
            model.cuda() 
            copy_state_dict(checkpoint['state_dict'], model)
            model_new=model
        else:
            model_new, model_last,  model_long= train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                                writer,logger_res=logger_res, model_last=model_last, model_long=model_long)
        if 0==set_index:
            model=model_new 
        elif 1==set_index:
            model_old=model_last # complete old knowledge
            best_alpha = get_adaptive_alpha(args, model_new, model_last, all_train_sets, set_index)
            model = linear_combination(args, model_new, model_last, best_alpha) # inference model
             
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)
        elif set_index>=2:
            # obtain optimal alpha
            best_alpha=search_alpha(model_long, model_last, all_train_sets[set_index][4],args)
            # fuse old models
            model_old = linear_combination(args, model_long, model_last, best_alpha)
            
            # generate inference model 
            best_alpha = get_adaptive_alpha(args, model_new, model_old, all_train_sets, set_index)
            model = linear_combination(args, model_new, model_old, best_alpha)   
            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[set_index], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))

            
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)
        # 模型
        model_last=copy.deepcopy(model_new)
        model_long=model_old

    print('finished')
def search_alpha(model_long, model_last, init_loader,args, Norm=10):
    res={}
    best_alpha=0.
    best_score=-1000000.
    for alpha in torch.arange(0,1.05, 0.1):
        model_fuse=linear_combination(args, model_long, model_last, alpha)

        features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named, vars_mean = extract_features_proto(model_fuse,
                                                                                                                  init_loader,
                                                                                                            get_mean_feature=True)  # init_loader is original designed for classifer init
        matric='mAP' # mAP sim
        if 'mAP' == matric:
            # if 'cu'
            try:
                map=fast_eval(features_all_old,labels_all_old, camids_all, args)
            except:
                camids_all=list(range(len(camids_all)))
                map=fast_eval(features_all_old,labels_all_old, camids_all, args)
            if map>best_score:
                best_score=map
                best_alpha=alpha
        else:
            from reid.metric_learning.distance import cosine_similarity
            features_all_old=torch.stack(features_all_old)
            labels_all_old=torch.tensor(labels_all_old).cuda()
            

            sim_matrix=cosine_similarity(features_all_old, features_all_old)   # 距离

            sim_matrix=-(torch.eye(len(sim_matrix))*100).cuda()+sim_matrix    # 对角线变为负无穷
            
            sim_matrix=F.softmax(sim_matrix*Norm, dim=1)
            GT=(labels_all_old.unsqueeze(0)-labels_all_old.unsqueeze(1))==0 # GT矩阵

            score=(sim_matrix*GT.float()-sim_matrix*(1-GT.float())).sum()
            res[alpha]=score

            if score>best_score:
                best_alpha=alpha
                best_score=score
    # model_fuse=linear_combination(args, model_long, model_last, best_alpha)
    print("####evaluating results####:", res)
    print("best alpha:",best_alpha, "best_value:",best_score)
    # return model_fuse
    return best_alpha
    

def get_normal_affinity(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_matrix_origin=-100*torch.eye(x.size(0)).to(x.device)+pre_matrix_origin
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_normal_affinity_origin(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    # pre_matrix_origin=-100*torch.eye(x.size(0)).to(x.device)+pre_matrix_origin
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]  # trainloader of current dataset
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new, args.global_alpha)
    Affin_old = get_normal_affinity(features_all_old, args.global_alpha)

   
    
    # transform similarity to the fusion weight
    sim=(Affin_new*Affin_old).sum(-1).mean()
    alpha=sim
    if args.absolute_delta:
        Affin_new = get_normal_affinity_origin(features_all_new, args.global_alpha)
        Affin_old = get_normal_affinity_origin(features_all_old, args.global_alpha)
        Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()
        alpha=float(1-Difference)
    return alpha

def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None,
                  model_last=None, model_long=None):
    
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains
    
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    
        # Initialize classifer with class centers    
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()
    if set_index>0:
        model_last.eval()   # 上一阶段训练模型
        if set_index>1:
            model_long.eval()   # t-2 EMA模型
    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(cfg, args, model, add_num + num_classes,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,
                      model_last=model_last, model_long=model_long
                      )
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test or epoch+1==Epochs:
                
                mAP = fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)                
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model, model_last, model_long


def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    print("*******combining the models with alpha: {}*******".format(alpha))
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = min(model_old_state_dict[k].shape[0], v.shape[0])
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k][:num_class_old]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x','vit'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xukunlun/DATA/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--resume_folder', type=str, default=None, help="test the models in a file")
    parser.add_argument('--resume_id',  type=int, default=-1,  help="")   


    parser.add_argument('--setting', type=int, default=1, choices=[1, 2, 51,52,53,54,55], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")   
    parser.add_argument('--tau', default=0.1, type=float, help="softmax temperture")  
    parser.add_argument('--weighted_loss', action='store_true', help="the new and old prototypes are used for global information!")
    
    parser.add_argument('--mse', action='store_true', help="using mse loss instead of KL loss")
    parser.add_argument('--mae', action='store_true', help="using mae loss instead of KL loss")
    parser.add_argument('--js', action='store_true', help="using js-diverigence loss instead of KL loss")

    parser.add_argument('--global_alpha',  type=float, default=400,  help="")      
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    parser.add_argument('--absolute_delta', action='store_true', help="only use dual teacher")
    main()
