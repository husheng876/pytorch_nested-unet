import torch
from torch import nn

######the header file from CRDN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['UNet', 'NestedUNet','UNetRNN','UNetRNNGhost']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

#在__init__中加入了deep_supervision=False，不加在train中如果存在deep_supervison部分会出现参数报错
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,**kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

class RDC(nn.Module):
    def __init__(self, hidden_dim, kernel_size, bias, decoder='LSTM'):
        """
        Recurrent Decoding Cell (RDC) module.
        :param hidden_dim:
        :param kernel_size: conv kernel size
        :param bias: if or not to add a bias term
        :param decoder: <name> [options: 'vanilla, LSTM, GRU']
        """
        super(RDC, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = 1   #, kernel_size // 2   #整除，为何padding的形式是这样的，是否需要更改成别的样子
        self.bias = bias
        self.decoder = decoder
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size,padding=self.padding, stride=1,bias=self.bias)#param1，2:input channels,output channels，数据形式是默认channel first
        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,padding=self.padding, stride=1,bias=self.bias)
        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size,padding=self.padding,stride=1,bias=self.bias)
        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size,stride=1,
                                      padding=self.padding, bias=self.bias)

    def forward(self, x_cur, h_pre, c_pre=None): #使h_pre和c_pre都与x_cur保持一致大小
        if self.decoder == "LSTM":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)#将输入进行上/下采样到给定的大小或scale_facotr
            #upsampling operation
            c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.lstm_catconv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)#将输入变量从dim纬度划分成self.hidden_dim份均块
            #four gate which decide whether or how much to propagate both semantic and spatial information to the next RDC
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_cur = f * c_pre_up + i * g    #indicate the cell state of ConvLSTM
            h_cur = o * torch.tanh(c_cur)   #hidden

            return h_cur, c_cur

        elif self.decoder == "GRU":
            #通常考虑pixels为square而不是点，align_corners是true时，输入输出张量以角像素的中心点对齐，保留角像素的值
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.gru_catconv(combined)
            cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv(torch.cat([x_cur, r * h_pre_up], dim=1)))
            h_cur = z * h_pre_up + (1 - z) * h_hat

            return h_cur

        elif self.decoder == "vanilla":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            #print("？？？？The shape of h_pre_up is ?????",np.shape(h_pre_up))
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            #print("？？？？The shape of combined is ?????", np.shape(combined))
            combined_conv = self.vanilla_conv(combined)
            #print("？？？？The shape of combined_conv is ?????", np.shape(combined_conv))
            h_cur = torch.relu(combined_conv)
            #print("************The output shape is***********:",np.shape(h_cur))
            return h_cur

"""
Implementation code for CRDN with U-Net-backbone (UNetRNN).
输入大小和输出图像大小一样
"""
class UNetRNN(nn.Module):
    def __init__(self, n_classes,input_channel=3,  kernel_size=3, feature_scale=4, decoder="vanilla", bias=True,deep_supervision=False,**kwargs):

        super(UNetRNN, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512,1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], is_batchnorm=True)#参 数2表示out_channel，函数不改变图像大小只改变通道数
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  #大小比原先的变小一半
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm=True)

        #this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),#5的卷积核大小
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias, 
                       decoder=self.decoder)

    def forward(self, input, cell_state=None):
        conv1 = self.conv1(input)  # 1,filters[0]   # 图像大小是1,输出通道是filters[0]

        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)  # 1/2,filters[1]

        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)  # 1/4,filters[2]

        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)  # 1/8,filters[3]

        maxpool4 = self.maxpool4(conv4)
        conv5 = self.center(maxpool4)  # 1/16,filters[4]

        x1 = self.score_block5(conv5)  # 图像大小1/16,输出通道是class
        #print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        #print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        #print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        #print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        #print("#####The input shape of x5 is:", np.shape(x5))

        h0 = self._init_cell_state(x1)  # 1/16,512       返回与x1大小相同的在cuda中的零张量
        #print("#####The input shape of h0 is:", np.shape(h0))  #(16,1,6,6)
        
        # Decode
        if self.decoder == "LSTM":
            # init c0
            if cell_state is not None:
                raise NotImplementedError()
            else:
                c0 = self._init_cell_state(h0)

            h1, c1 = self.RDC(x_cur=x1, h_pre=h0, c_pre=c0)  # 1/16,class
            h2, c2 = self.RDC(x_cur=x2, h_pre=h1, c_pre=c1)  # 1/8,class
            h3, c3 = self.RDC(x_cur=x3, h_pre=h2, c_pre=c2)  # 1/4,class
            h4, c4 = self.RDC(x_cur=x4, h_pre=h3, c_pre=c3)  # 1/2,class
            h5, c5 = self.RDC(x_cur=x5, h_pre=h4, c_pre=c4)  # 1,class

        elif self.decoder == "GRU":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        elif self.decoder == "vanilla":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        else:
            raise NotImplementedError

        return h5

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size()).cuda(0)

#在UNetRNN中有实现，用于downsampling的卷积
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )#参数3，4分别是kernel_size和stride，BatchNorm2d对out_size进行数据归一化操作
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )#Conv2d参数4,5是stride，padding
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

#######################################
#the model of UnetRNN which is composed with Ghost and UNetRNN
#######################################

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)   #返回数字的上入整数
        new_channels = init_channels * (ratio - 1)   #此时=init_channels数

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        print("The output of the model GhostModule is :",out[:, :self.oup, :, :])
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)#mid_chs参数表示GhostModule模块的输出channel

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),  #不改变图像大小
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),#不改变图像大小
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

class UNetRNNGhost(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNGhost, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], is_batchnorm=True)  # 参 数2表示out_channel，函数不改变图像大小只改变通道数
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 大小比原先的变小一半
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm=True)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = unetConv2(filters[3], filters[4], is_batchnorm=True)

        # this block output is ghost block from paper "Segmenting Medical MRI via Recurrent Decoding Cell"
        self.score_block1 = nn.Sequential(GhostBottleneck(in_chs=filters[0],mid_chs=filters[0]//2,out_chs=self.n_classes))

        self.score_block2 = nn.Sequential(GhostBottleneck(in_chs=filters[1],mid_chs=filters[1]//2,out_chs=self.n_classes))

        self.score_block3 = nn.Sequential(GhostBottleneck(in_chs=filters[2],mid_chs=filters[2]//2,out_chs=self.n_classes))

        self.score_block4 = nn.Sequential(GhostBottleneck(in_chs=filters[3],mid_chs=filters[3]//2,out_chs=self.n_classes))

        self.score_block5 = nn.Sequential(GhostBottleneck(in_chs=filters[4],mid_chs=filters[4]//2,out_chs=self.n_classes))

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias,
                       decoder=self.decoder)

    def forward(self, input, cell_state=None):
        conv1 = self.conv1(input)  # 1,filters[0]   # 图像大小是1,输出通道是filters[0]

        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)  # 1/2,filters[1]

        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)  # 1/4,filters[2]

        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)  # 1/8,filters[3]

        maxpool4 = self.maxpool4(conv4)
        conv5 = self.center(maxpool4)  # 1/16,filters[4]

        x1 = self.score_block5(conv5)  # 图像大小1/16,输出通道是class
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        # print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        # print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        # print("#####The input shape of x5 is:", np.shape(x5))

        h0 = self._init_cell_state(x1)  # 1/16,512       返回与x1大小相同的在cuda中的零张量
        # print("#####The input shape of h0 is:", np.shape(h0))  #(16,1,6,6)

        # Decode
        if self.decoder == "LSTM":
            # init c0
            if cell_state is not None:
                raise NotImplementedError()
            else:
                c0 = self._init_cell_state(h0)

            h1, c1 = self.RDC(x_cur=x1, h_pre=h0, c_pre=c0)  # 1/16,class
            h2, c2 = self.RDC(x_cur=x2, h_pre=h1, c_pre=c1)  # 1/8,class
            h3, c3 = self.RDC(x_cur=x3, h_pre=h2, c_pre=c2)  # 1/4,class
            h4, c4 = self.RDC(x_cur=x4, h_pre=h3, c_pre=c3)  # 1/2,class
            h5, c5 = self.RDC(x_cur=x5, h_pre=h4, c_pre=c4)  # 1,class

        elif self.decoder == "GRU":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        elif self.decoder == "vanilla":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)  # 1,class

        else:
            raise NotImplementedError

        return h5

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size()).cuda(0)