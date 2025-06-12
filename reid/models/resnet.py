import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import math
from reid.models.gem_pool import GeneralizedMeanPoolingP
from torch.nn import functional as F
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID
class Backbone(nn.Module):
    def __init__(self,last_stride, bn_norm, with_ibn, with_se,block, num_classes,layers):
        super(Backbone, self).__init__()
        self.in_planes = 2048
        self.base = ResNet(last_stride=last_stride,
                            block=block,
                            layers=layers)
        print('using resnet50 as a backbone')

        
        self.bottleneck = nn.BatchNorm2d(2048)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = nn.Linear(512*block.expansion, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
       

        self.random_init()
        self.num_classes = num_classes
    def forward_head(self,x):
        x=x.unsqueeze(-1).unsqueeze(-1)
        # print(x.shape)
        bottle=copy.deepcopy(self.bottleneck)
        bottle.eval()
        bn_feat = self.bottleneck(x)
        return self.classifier(bn_feat[...,0,0])


    def forward(self, x, domains=None, training_phase=None, get_all_feat=False,epoch=0, head_only=False):     
        if head_only:
            return self.forward_head(x)   
        x = self.base(x)
        global_feat = self.pooling_layer(x) # [16, 2048, 1, 1]
        bn_feat = self.bottleneck(global_feat) # [16, 2048, 1, 1]
        
        # global_feat=F.normalize(global_feat)    # L2 normalization
            

        if get_all_feat is True:
            cls_outputs = self.classifier(bn_feat[..., 0, 0])
            return global_feat[..., 0, 0], bn_feat[..., 0, 0], cls_outputs, x

        if self.training is False:
            # return bn_feat[..., 0, 0]
            return global_feat[..., 0, 0]

        bn_feat = bn_feat[..., 0, 0]
        cls_outputs = self.classifier(bn_feat)      
        return global_feat[..., 0, 0], bn_feat, cls_outputs, x

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class vision_transformer(nn.Module):
    def __init__(self, num_classes):
        super(vision_transformer, self).__init__()
        self.in_planes = 768
        self.base = vit_base_patch16_224_TransReID(img_size=[256,128],camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1)
        self.base.load_param()
        print('using vison transformer as a backbone')

        
        self.bottleneck = nn.BatchNorm2d(768)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        # self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
       

        self.random_init()
        self.num_classes = num_classes
    def forward(self, x, domains=None, training_phase=None, get_all_feat=False,epoch=0):      
        Shape=x.shape  
        x = self.base(x)
        global_feat = x[:,0] # [16, 2048, 1, 1]
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        x=x[:,1:].permute(0,2,1)
        x=x.reshape(Shape[0],-1,Shape[2]//16, Shape[3]//16)
        bn_feat = self.bottleneck(global_feat) # [16, 2048, 1, 1]
        
        # global_feat=F.normalize(global_feat)    # L2 normalization
            

        if get_all_feat is True:
            cls_outputs = self.classifier(bn_feat[..., 0, 0])
            return global_feat[..., 0, 0], bn_feat[..., 0, 0], cls_outputs, x

        if self.training is False:
            # return bn_feat[..., 0, 0]
            return global_feat[..., 0, 0]

        bn_feat = bn_feat[..., 0, 0]
        cls_outputs = self.classifier(bn_feat)      
        return global_feat[..., 0, 0], bn_feat, cls_outputs, x

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def make_model(arg, num_class, camera_num, view_num,pretrain=True):
    if '50x'==arg.MODEL:
        model = Backbone(1, 'BN', False, False, Bottleneck, num_class, [3, 4, 6, 3])
        print('===========building ResNet===========')
        if pretrain:
            import torchvision
            res_base = torchvision.models.resnet50(pretrained=True)
            res_base_dict = res_base.state_dict()

            state_dict = model.base.state_dict()
            for k, v in res_base_dict.items():
                if k in state_dict:
                    if v.shape == state_dict[k].shape:
                        state_dict[k] = v
                    else:
                        print('param {} of shape {} does not match loaded shape {}'.format(k, v.shape,
                                                                                        state_dict[k].shape))
                else:
                    print('param {} in pre-trained model does not exist in this model.base'.format(k))

            model.base.load_state_dict(state_dict, strict=True)
    elif 'vit'==arg.MODEL:
        model=vision_transformer(num_class)
        # model =vit_base_patch16_224_TransReID(img_size=[256,128],camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1)


    
    
    return model
