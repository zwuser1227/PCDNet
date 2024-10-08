import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
import cv2 as cv
import os
from PIL import Image, ImageStat
from GaborLayer import GaborConv, GaborConvSca, GaborConv2d, GaborYu
import torchvision
class attebtion_module(nn.Module):
    def __init__(self, in_channels, out_channels,nScale):
        super(attebtion_module, self).__init__()
 
        self.stages = GaborConv(in_channels, out_channels, 3, padding=1, stride=1, bias=False, groups=1, expand=False, nScale=1)
        # self.stages = self.make_conv(in_channels, [out_channels], padding=1, stride=1, rate=1)
        self.stage = self.make_conv(out_channels, [out_channels], padding= 1, stride = 1, rate= 1)

        self.attention_out = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        x_Gabor = self.stages(x)
        x_Gabor = F.relu(x_Gabor)
        x1 = self.stage(x_Gabor)
        attention = self.attention_out(x1)
        attention_weight = attention.sigmoid()

        attention_out = x1 * attention_weight

        return self.conv1(self.conv3(x_Gabor) + attention_out)
        # return x1
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_conv(in_channels, cfg, padding=1, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [conv2d]
                in_channels = v
        return nn.Sequential(*layers)
class new_backnone(nn.Module):
    def __init__(self):
        super(new_backnone, self).__init__()
        #数字卷积核个数，M代表池化层
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        #VGG16的13个卷积层

        self.stage_1 = self.make_layers(1, [8, 16])
        # self.smish= Smish()#nn.ReLU(inplace=True)
        self.stage_2 = self.make_layers(1, [8, 16])

        self.stage_3 = self.make_layers(1, [8, 16])
        self.dilated_conv2d_1 = self.make_layers(1, [8], padding=2, stride=1, rate=2)
        self.dilated_conv2d = self.make_layers(8, [16], padding=4, stride=1, rate=4)

        self.stage1 = attebtion_module(16, 16, nScale=1)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage2 = attebtion_module(16, 32, nScale=2)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage3 = attebtion_module(32, 64, nScale=3)
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage4 = attebtion_module(64, 128, nScale=4)
    #正向传播过程
    # def forward(self, x, feedx):
    def forward(self, x):
        x1 = x[:, 0:1, :, :]  
        x2 = x[:, 1:2, :, :]  
        x3 = x[:, 2:3, :, :]  
        gray = (x1 + x2 + x2) / 3
        stage_x1 = self.stage_1(x1)
        stage_x2 = self.stage_2(x2)
        stage_x3 = self.stage_3(x3)

        fusion = stage_x1 + stage_x2 + stage_x3
        dilated_gray1 = self.dilated_conv2d_1(gray)
        dilated_gray = self.dilated_conv2d(dilated_gray1)

        input = fusion + dilated_gray

        stage1 = self.stage1(input)
        pooling2 = self.pooling2(stage1)

        stage2 = self.stage2(pooling2)
        pooling3 = self.pooling3(stage2)

        stage3 = self.stage3(pooling3)
        pooling4 = self.pooling4(stage3)

        stage4 = self.stage4(pooling4)

        return stage1, stage2, stage3, stage4

        # return stage1, stage2, stage3, stage4
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, padding=1, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                # layers += [conv2d]
                in_channels = v
        return nn.Sequential(*layers)
    #权重初始化函数
    def _initialize_weights(self, dict_path):
        model_paramters = torch.load(dict_path)#读取的参数
        # print(model_paramters)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = (model_paramters.popitem(last=False)[-1])
                m.bias.data = model_paramters.popitem(last=False)[-1]
#权重卷积块对应不同的流，其中一个使用sigmod函数激活，再融合

from utilss.AF.Xsmish import Smish
#权重卷积块对应不同的流，其中一个使用sigmod函数激活，再融合
class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels, D, groups):
        super(adap_conv, self).__init__()
        # self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #                             nn.BatchNorm2d(out_channels),
        #                             nn.ReLU(inplace=True)])
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)])
        self.smish= Smish()#nn.ReLU(inplace=True)
       
    def forward(self, x):
        x = self.conv(x)
        x= self.smish(x)
       
        return x

class adap_conv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adap_conv1, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
      
    def forward(self, x):
        x = self.conv(x)
    
        return x

#特征图融合，细化块把低分辨率特征图上采样高分辨率图
class Refine_block2(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2, self).__init__()
        # self.pre_conv1 = adap_conv1(in_channel[0], out_channel)
        self.pre_conv2 = adap_conv1(in_channel[1], out_channel)
        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
        # self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, *input):
        # x1 = self.pre_conv1(input[0])
        x1 = input[0]
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x2


class conv_d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_d, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.dilated_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.attention = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
        # self.conv0 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smish= Smish()#nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        # xo =x
        x = self.dilated_conv(x)
        x = self.smish(x)
        # x =  F.relu(x)
        attention = self.attention(x)
        attention_weight = attention.sigmoid()
        sg = torch.mean(attention_weight)
        return x * attention_weight + sg * x
        # return xo
#特征图融合，细化块把低分辨率特征图上采样高分辨率图
class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, D, groups, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel, D, groups)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel, D, groups)

        # self.Relu = nn.ReLU(inplace=True)
        self.attention = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)

        self.smish= Smish()#nn.ReLU(inplace=True)
        # self.weight1 = nn.Parameter(torch.Tensor([0.]))
        # self.weight2 = nn.Parameter(torch.Tensor([0.]))

        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
        # self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])


        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2
class wave_attention(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(wave_attention, self).__init__()
        self.pre_conv1 = conv_d(in_channel[0], out_channel)
        self.pre_conv2 = conv_d(in_channel[1], out_channel)
        self.attention = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)

        # self.smish= Smish()#nn.ReLU(inplace=True)
        # self.weight1 = nn.Parameter(torch.Tensor([0.]))
        # self.weight2 = nn.Parameter(torch.Tensor([0.]))

        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
        # self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        # attention1 = self.attention(x2)
        # attention2 = self.attention(x2)

        return x1 + x2
class super_pixels(nn.Module):
    def __init__(self, inplanes, factor):
        super(super_pixels, self).__init__()
        self.superpixels = nn.PixelShuffle(factor) #伸缩
        planes = int(inplanes/(factor*2))
        self.down_sample = nn.Conv2d(planes, 1, kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        x = self.superpixels(x)
        x = self.down_sample(x)
        return x

class Pool_Conv_no_change(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Pool_Conv_no_change, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage_pool = self.make_layers(in_channel, [out_channel])
        self.stage_no_change = self.make_layers_1(in_channel, [in_channel])
    def forward(self, input):
        pool = self.pool(input)
        stage_pool = self.stage_pool(pool)            #与下一层3*3不变的sum
        stage_no_change = self.stage_no_change(input) #3*3不变
        stage_no_with_1 = input                       #不处理，与下一层上采样的融合
        stage_no_with_2 = input                       #上采样与上一层融合
        return stage_no_with_1, stage_no_with_2, stage_pool, stage_no_change


#new_backbone的解码，四个输出
class new_decode(nn.Module):
    def __init__(self):
        super(new_decode, self).__init__()
        #融合
        self.level2_11 = wave_attention((16, 32), 16, 2)
        self.level2_22 = wave_attention((32, 64), 32, 2)
        self.level2_33 = wave_attention((64, 128), 64, 2)
        # self.level2_44 = Refine_block2_1((512, 512), 512, 2, 32, 16)
        #融合
        self.level3_11 = wave_attention((16, 32), 16, 2)
        self.level3_22 = wave_attention((32, 64), 32, 2)
        # self.level3_33 = Refine_block2_1((256, 512), 256, 2, 32, 16)
#融合
        self.level4_1 = wave_attention((16, 32), 16, 2)
        # self.level4_2 = Refine_block2_1((128, 256), 128, 2, 32, 16)

        # self.level5_1 = Refine_block2_1((64, 128), 64, 2, 32, 16)

        self.level7 = nn.Conv2d(16, 1, kernel_size=1, padding=0)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):
        level2_11 = self.level2_11(input[0], input[1])
        level2_22 = self.level2_22(input[1], input[2])
        level2_33 = self.level2_33(input[2], input[3])
        # level2_44 = self.level2_44(input[3], input[4])

        level3_11 = self.level3_11(level2_11, level2_22)
        level3_22 = self.level3_22(level2_22, level2_33)
        # level3_33 = self.level3_33(level2_33, level2_44)

        level4_1 = self.level4_1(level3_11, level3_22)
        # level4_2 = self.level4_2(level3_22, level3_33)

        # level5 = self.level5_1(level4_1)
        return self.level7(level4_1)


class PCDNet(nn.Module):
    def __init__(self):
        super(PCDNet, self).__init__()
        self.light_dark_module = light_dark_module()
        # self.feed_new_decode = new_backnone1()
        self.encode = new_backnone()
        self.decode1 = new_decode()
        # self.v1 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
    def forward(self, x):
        # light_dark_result =x
        light_dark_result = self.light_dark_module(x)
        # feed_new_decode = self.feed_new_decode(light_dark_result)
        # end_points = self.encode(light_dark_result, feed_new_decode)  # 输出特征
        end_points = self.encode(light_dark_result)  # 输出特征
        x = self.decode1(*end_points)  # 把输出的特征进行反卷积解码
        # v = self.v1(end_points[4])
        return x, light_dark_result