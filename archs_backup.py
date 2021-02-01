#################################
#本文件用于存储archs中调试好的结构代码，用于简化archs代码长度
#当需要使用相关结构模型时
#################################
import torch
from torch import nn

######the header file from CRDN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import *

#UNetRNNCAttention_PSP  模型用到的引入
from segmentation_refinement.models.psp.pspnet import *

#from cascadePSP_model.sync_batchnorm import *
#from cascadePSP_model.psp.pspnet import *
__all__ = ['UNet', 'NestedUNet','UNetRNN','UNetRNNGhost','UNetRM3','UNetRM7','UNetRNNPAttention',
           'UNetRNNCAttention','UNetRNNAttention','UNetRNNCAttention_PSP','UNetRNNPSP','R2U_Net']

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


# 在__init__中加入了deep_supervision=False，不加在train中如果存在deep_supervison部分会出现参数报错
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

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

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

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
        self.padding = 1  # , kernel_size // 2   #整除，为何padding的形式是这样的，是否需要更改成别的样子
        self.bias = bias
        self.decoder = decoder
        self.gru_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, self.kernel_size, padding=self.padding,
                                     stride=1,
                                     bias=self.bias)  # param1，2:input channels,output channels，数据形式是默认channel first
        self.gru_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size, padding=self.padding,
                                  stride=1, bias=self.bias)
        self.lstm_catconv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, self.kernel_size, padding=self.padding,
                                      stride=1, bias=self.bias)
        self.vanilla_conv = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, self.kernel_size, stride=1,
                                      padding=self.padding, bias=self.bias)

    def forward(self, x_cur, h_pre, c_pre=None):  # 使h_pre和c_pre都与x_cur保持一致大小
        if self.decoder == "LSTM":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear',
                                     align_corners=True)  # 将输入进行上/下采样到给定的大小或scale_facotr
            # upsampling operation
            c_pre_up = F.interpolate(c_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            combined = torch.cat([h_pre_up, x_cur], dim=1)
            combined_conv = self.lstm_catconv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
                                                 dim=1)  # 将输入变量从dim纬度划分成self.hidden_dim份均块
            # four gate which decide whether or how much to propagate both semantic and spatial information to the next RDC
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)

            c_cur = f * c_pre_up + i * g  # indicate the cell state of ConvLSTM
            h_cur = o * torch.tanh(c_cur)  # hidden

            return h_cur, c_cur

        elif self.decoder == "GRU":
            # 通常考虑pixels为square而不是点，align_corners是true时，输入输出张量以角像素的中心点对齐，保留角像素的值
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)

            combined = torch.cat([h_pre_up.cuda(), x_cur.cuda()], dim=1)
            combined_conv = self.gru_catconv(combined)
            # combined_conv = combined_conv.cuda()
            cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
            r = torch.sigmoid(cc_r)
            z = torch.sigmoid(cc_z)
            h_hat = torch.tanh(self.gru_conv(torch.cat([x_cur, r * h_pre_up], dim=1)))
            h_cur = z * h_pre_up + (1 - z) * h_hat

            return h_cur

        elif self.decoder == "vanilla":
            h_pre_up = F.interpolate(h_pre, size=[x_cur.size(2), x_cur.size(3)], mode='bilinear', align_corners=True)
            # print("？？？？The shape of h_pre_up is ?????",np.shape(h_pre_up))
            combined = torch.cat([h_pre_up.cpu(), x_cur.cpu()], dim=1)  # cpu是后面加的
            # print("？？？？The shape of combined is ?????", np.shape(combined))
            combined_conv = self.vanilla_conv(combined)
            # print("？？？？The shape of combined_conv is ?????", np.shape(combined_conv))
            h_cur = torch.relu(combined_conv)
            # print("************The output shape is***********:",np.shape(h_cur))
            return h_cur


"""
Implementation code for CRDN with U-Net-backbone (UNetRNN).
输入大小和输出图像大小一样
"""


class UNetRNN(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNN, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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


# 在UNetRNN中有实现，用于downsampling的卷积
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )  # 参数3，4分别是kernel_size和stride，BatchNorm2d对out_size进行数据归一化操作
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )  # Conv2d参数4,5是stride，padding
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


#######################################
# the model of UnetRNN which is composed with Ghost and UNetRNN
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
        init_channels = math.ceil(oup / ratio)  # 返回数字的上入整数
        new_channels = init_channels * (ratio - 1)  # 此时=init_channels数

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
        # print("The output of the model GhostModule is :",out[:, :self.oup, :, :])
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)  # mid_chs参数表示GhostModule模块的输出channel

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
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),  # 不改变图像大小
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),  # 不改变图像大小
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

        filters = [64, 128, 256, 512, 1024]
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
        self.score_block1 = nn.Sequential(
            GhostBottleneck(in_chs=filters[0], mid_chs=filters[0] // 2, out_chs=self.n_classes))

        self.score_block2 = nn.Sequential(
            GhostBottleneck(in_chs=filters[1], mid_chs=filters[1] // 2, out_chs=self.n_classes))

        self.score_block3 = nn.Sequential(
            GhostBottleneck(in_chs=filters[2], mid_chs=filters[2] // 2, out_chs=self.n_classes))

        self.score_block4 = nn.Sequential(
            GhostBottleneck(in_chs=filters[3], mid_chs=filters[3] // 2, out_chs=self.n_classes))

        self.score_block5 = nn.Sequential(
            GhostBottleneck(in_chs=filters[4], mid_chs=filters[4] // 2, out_chs=self.n_classes))

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

#########################################
# the code to short the convolution block number of UNetRNN,the channel is:64,288,512
#########################################
class UNetRM3(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRM3, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 288, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0], is_batchnorm=True)  # 参 数2表示out_channel，函数不改变图像大小只改变通道数

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 大小比原先的变小一半
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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

        self.RDC = RDC(self.n_classes, self.kernel_size, bias=self.bias,
                       decoder=self.decoder)

    def forward(self, input, cell_state=None):
        conv1 = self.conv1(input)  # 1,filters[0]   # 图像大小是1,输出通道是filters[0]

        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)  # 1/2,filters[1]

        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)  # 1/4,filters[2]

        x1 = self.score_block3(conv3)  # 图像大小1/16,输出通道是class
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block2(conv2)  # 1/8,class
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block1(conv1)  # 1/4,class
        # print("#####The input shape of x3 is:", np.shape(x3))

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

        elif self.decoder == "GRU":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class

        elif self.decoder == "vanilla":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class

        else:
            raise NotImplementedError

        return h3

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size()).cuda(0)

class UNetRM7(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRM7, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [32, 64, 128, 256, 512, 1024, 2048]  # 原为64, 128, 256, 512,1024
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.input_channel, filters[0],
                               is_batchnorm=True)  # 参 数2表示out_channel，函数不改变图像大小只改变通道数
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 大小比原先的变小一半
        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm=True)  # 16channel,32

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm=True)  # 32.64

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm=True)  # 64,128

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = unetConv2(filters[3], filters[4], is_batchnorm=True)  # 128,256

        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = unetConv2(filters[4], filters[5], is_batchnorm=True)  # 256,512

        self.maxpool6 = nn.MaxPool2d(kernel_size=2)
        self.conv7 = unetConv2(filters[5], filters[6], is_batchnorm=True)  # 512,1024

        # this block output is cell current map
        self.score_block1 = nn.Sequential(
            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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

        self.score_block6 = nn.Sequential(
            nn.Conv2d(filters[5], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block7 = nn.Sequential(
            nn.Conv2d(filters[6], self.n_classes, 5, padding=2),
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
        conv5 = self.conv5(maxpool4)  # 1/8,filters[3]  #期望有32通道

        maxpool5 = self.maxpool5(conv5)
        conv6 = self.conv6(maxpool5)  # 1/8,filters[3]

        maxpool6 = self.maxpool6(conv6)
        conv7 = self.conv7(maxpool6)  # 1/8,filters[3]

        x1 = self.score_block7(conv7)  # 图像大小1/16,输出通道是class
        x2 = self.score_block6(conv6)  # 1/8,class
        x3 = self.score_block5(conv5)  # 1/4,class
        x4 = self.score_block4(conv4)  # 1/2,class
        x5 = self.score_block3(conv3)
        x6 = self.score_block2(conv2)
        x7 = self.score_block1(conv1)

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
            h5, c5 = self.RDC(x_cur=x5, h_pre=h4, c_pre=c4)
            h6, c6 = self.RDC(x_cur=x6, h_pre=h5, c_pre=c5)
            h7, c7 = self.RDC(x_cur=x7, h_pre=h6, c_pre=c6)

        elif self.decoder == "GRU":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)
            h6 = self.RDC(x_cur=x6, h_pre=h5)
            h7 = self.RDC(x_cur=x7, h_pre=h6)

        elif self.decoder == "vanilla":
            h1 = self.RDC(x_cur=x1, h_pre=h0)  # 1/16,class
            h2 = self.RDC(x_cur=x2, h_pre=h1)  # 1/8,class
            h3 = self.RDC(x_cur=x3, h_pre=h2)  # 1/4,class
            h4 = self.RDC(x_cur=x4, h_pre=h3)  # 1/2,class
            h5 = self.RDC(x_cur=x5, h_pre=h4)
            h6 = self.RDC(x_cur=x6, h_pre=h5)
            h7 = self.RDC(x_cur=x7, h_pre=h6)

        else:
            raise NotImplementedError

        return h7

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size()).cuda(0)

###############################
# The position attention module and channel attention module from the paper "Dual Attention Network for Scene Segmentation"
###############################
class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # self.sizeModule = 0

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        # print(out.size(), out.type())
        # self.sizeModule=out.size()#自己加的语句
        # print("The value of sizeModule is:",self.sizeModule)
        return out


class Attention_block(nn.Module):
    def __init__(self, in_dim):
        super(Attention_block, self).__init__()
        self.channel_in = in_dim

    def forward(self, input):
        pa = PAM_Module(self.channel_in)  # position attention module
        ca = CAM_Module(self.channel_in)  # channel attention module
        pa = pa(input)
        ca = ca(input)
        out = pa + ca

        return out


############################
# The architecture of UNetRNNAttenion is compose with UNetRNN and the function of PAM_Module and CAM_Module
############################
class UNetRNNPAttention(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNPAttention, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        self.PAM_Module1 = PAM_Module(filters[0])
        self.PAM_Module2 = PAM_Module(filters[1])
        self.PAM_Module3 = PAM_Module(filters[2])
        self.PAM_Module4 = PAM_Module(filters[3])
        self.PAM_Module5 = PAM_Module(filters[4])

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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
        x1 = self.PAM_Module1(x1)
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        x2 = self.PAM_Module2(x2)
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        x3 = self.PAM_Module3(x3)
        # print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        x4 = self.PAM_Module4(x4)
        # print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        x5 = self.PAM_Module5(x5)
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


class UNetRNNCAttention(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNCAttention, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        self.CAM_Module1 = CAM_Module(filters[4])
        self.CAM_Module2 = CAM_Module(filters[3])
        self.CAM_Module3 = CAM_Module(filters[2])
        self.CAM_Module4 = CAM_Module(filters[1])
        self.CAM_Module5 = CAM_Module(filters[0])

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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
        x1 = self.CAM_Module1(x1)  # x1类型<class 'archs.CAM_Module'>
        # print("The value of CAM_Module is:",x1)  ##CAM_Module((softmax): Softmax())
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        x2 = self.CAM_Module2(x2)
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        x3 = self.CAM_Module3(x3)
        # print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        x4 = self.CAM_Module4(x4)
        # print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        x5 = self.CAM_Module5(x5)
        # print("#####The input shape of x5 is:", np.shape(x5))
        # print("The type of x1", type(x1))  #<class 'archs.CAM_Module'>
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
        # print("The size of the Channle attention module is:",tensor)  #CAM_Module((softmax): Softmax())
        # print("The type of the Channle attention module is:", type(tensor))#<class 'archs.CAM_Module'>
        return torch.zeros(tensor.size()).cuda(0)


################################################
# the moduel with attention block that compose with the
################################################
class UNetRNNAttention(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNAttention, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        self.attention_block1 = Attention_block(filters[4])
        self.attention_block2 = Attention_block(filters[3])
        self.attention_block3 = Attention_block(filters[2])
        self.attention_block4 = Attention_block(filters[1])
        self.attention_block5 = Attention_block(filters[0])

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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
        x1 = self.attention_block1(x1)
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        x2 = self.attention_block2(x2)
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        x3 = self.attention_block3(x3)
        # print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        x4 = self.attention_block4(x4)
        # print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        x5 = self.attention_block5(x5)
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


class UNetRNNCAttention_PSP(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNCAttention_PSP, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        self.CAM_Module1 = CAM_Module(filters[4])
        self.CAM_Module2 = CAM_Module(filters[3])
        self.CAM_Module3 = CAM_Module(filters[2])
        self.CAM_Module4 = CAM_Module(filters[1])
        self.CAM_Module5 = CAM_Module(filters[0])

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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
        x1 = self.CAM_Module1(x1)  # x1类型<class 'archs.CAM_Module'>
        # print("The value of CAM_Module is:",x1)  ##CAM_Module((softmax): Softmax())
        # print("#####The input shape of x1 is:",np.shape(x1))   #(16,1,6,6)
        x2 = self.score_block4(conv4)  # 1/8,class
        x2 = self.CAM_Module2(x2)
        # print("#####The input shape of x2 is:", np.shape(x2))     #(16,1,12,12)
        x3 = self.score_block3(conv3)  # 1/4,class
        x3 = self.CAM_Module3(x3)
        # print("#####The input shape of x3 is:", np.shape(x3))
        x4 = self.score_block2(conv2)  # 1/2,class
        x4 = self.CAM_Module4(x4)
        # print("#####The input shape of x4 is:", np.shape(x4))
        x5 = self.score_block1(conv1)  # 1,class
        x5 = self.CAM_Module5(x5)
        # print("#####The input shape of x5 is:", np.shape(x5))
        # print("The type of x1", type(x1))  #<class 'archs.CAM_Module'>
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

        model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024,
                       backend='resnet50').cuda()  # 另此模型权重为cuda类型
        # print("The type of input is :",type(input))
        # print("The type of h5 is :", type(h5))
        output = model(input, h5)

        return output['pred_224']

    def _init_cell_state(self, tensor):
        # print("The size of the Channle attention module is:",tensor)  #CAM_Module((softmax): Softmax())
        # print("The type of the Channle attention module is:", type(tensor))#<class 'archs.CAM_Module'>
        return torch.zeros(tensor.size()).cuda(0)


###########################
# Define a module to refine the segmentation
###########################
def resize_max_side(im, size, method):
    h, w = im.shape[-2:]
    max_side = max(h, w)
    ratio = size / max_side
    if method in ['bilinear', 'bicubic']:
        return F.interpolate(im, scale_factor=ratio, mode=method, align_corners=False)
    else:
        return F.interpolate(im, scale_factor=ratio, mode=method)


def safe_forward(model, im, seg, inter_s8=None, inter_s4=None):
    """
    Slightly pads the input image such that its length is a multiple of 8
    """
    b, _, ph, pw = seg.shape
    if (ph % 8 != 0) or (pw % 8 != 0):
        newH = ((ph // 8 + 1) * 8)
        newW = ((pw // 8 + 1) * 8)
        p_im = torch.zeros(b, 3, newH, newW, device=im.device)
        p_seg = torch.zeros(b, 1, newH, newW, device=im.device) - 1

        p_im[:, :, 0:ph, 0:pw] = im
        p_seg[:, :, 0:ph, 0:pw] = seg
        im = p_im
        seg = p_seg

        if inter_s8 is not None:
            p_inter_s8 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
            p_inter_s8[:, :, 0:ph, 0:pw] = inter_s8
            inter_s8 = p_inter_s8
        if inter_s4 is not None:
            p_inter_s4 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
            p_inter_s4[:, :, 0:ph, 0:pw] = inter_s4
            inter_s4 = p_inter_s4

    images = model(im, seg, inter_s8, inter_s4)
    return_im = {}

    for key in ['pred_224', 'pred_28_3', 'pred_56_2']:
        return_im[key] = images[key][:, :, 0:ph, 0:pw]
    del images

    return return_im


def process_high_res_im(model, im, seg, L=900):
    stride = L // 2

    _, _, h, w = seg.shape

    """
    Global Step
    """
    if max(h, w) > L:
        im_small = resize_max_side(im, L, 'area')
        seg_small = resize_max_side(seg, L, 'area')
    elif max(h, w) < L:
        im_small = resize_max_side(im, L, 'bicubic')
        seg_small = resize_max_side(seg, L, 'bilinear')
    else:
        im_small = im
        seg_small = seg

    images = safe_forward(model, im_small, seg_small)

    pred_224 = images['pred_224']
    pred_56 = images['pred_56_2']

    """
    Local step
    """

    for new_size in [max(h, w)]:
        im_small = resize_max_side(im, new_size, 'area')
        seg_small = resize_max_side(seg, new_size, 'area')
        _, _, h, w = seg_small.shape

        combined_224 = torch.zeros_like(seg_small)
        combined_weight = torch.zeros_like(seg_small)

        r_pred_224 = (F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float() * 2 - 1
        r_pred_56 = F.interpolate(pred_56, size=(h, w), mode='bilinear', align_corners=False) * 2 - 1

        padding = 16
        step_size = stride - padding * 2
        step_len = L

        used_start_idx = {}
        for x_idx in range((w) // step_size + 1):
            for y_idx in range((h) // step_size + 1):

                start_x = x_idx * step_size
                start_y = y_idx * step_size
                end_x = start_x + step_len
                end_y = start_y + step_len

                # Shift when required
                if end_y > h:
                    end_y = h
                    start_y = h - step_len
                if end_x > w:
                    end_x = w
                    start_x = w - step_len

                # Bound x/y range
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)

                # The same crop might appear twice due to bounding/shifting
                start_idx = start_y * w + start_x
                if start_idx in used_start_idx:
                    continue
                else:
                    used_start_idx[start_idx] = True

                # Take crop
                im_part = im_small[:, :, start_y:end_y, start_x:end_x]
                seg_224_part = r_pred_224[:, :, start_y:end_y, start_x:end_x]
                seg_56_part = r_pred_56[:, :, start_y:end_y, start_x:end_x]

                # Skip when it is not an interesting crop anyway
                seg_part_norm = (seg_224_part > 0).float()
                high_thres = 0.9
                low_thres = 0.1
                if (seg_part_norm.mean() > high_thres) or (seg_part_norm.mean() < low_thres):
                    continue
                grid_images = safe_forward(model, im_part, seg_224_part, seg_56_part)
                grid_pred_224 = grid_images['pred_224']

                # Padding
                pred_sx = pred_sy = 0
                pred_ex = step_len
                pred_ey = step_len

                if start_x != 0:
                    start_x += padding
                    pred_sx += padding
                if start_y != 0:
                    start_y += padding
                    pred_sy += padding
                if end_x != w:
                    end_x -= padding
                    pred_ex -= padding
                if end_y != h:
                    end_y -= padding
                    pred_ey -= padding

                combined_224[:, :, start_y:end_y, start_x:end_x] += grid_pred_224[:, :, pred_sy:pred_ey,
                                                                    pred_sx:pred_ex]

                del grid_pred_224

                # Used for averaging
                combined_weight[:, :, start_y:end_y, start_x:end_x] += 1

        # Final full resolution output
        seg_norm = (r_pred_224 / 2 + 0.5)
        pred_224 = combined_224 / combined_weight
        pred_224 = torch.where(combined_weight == 0, seg_norm, pred_224)

    _, _, h, w = seg.shape
    images = {}
    images['pred_224'] = F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=True)

    return images['pred_224']


class UNetRNNPSP(nn.Module):
    def __init__(self, n_classes, input_channel=3, kernel_size=3, feature_scale=4, decoder="GRU", bias=True,
                 deep_supervision=False, **kwargs):

        super(UNetRNNPSP, self).__init__()
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.bias = bias

        filters = [64, 128, 256, 512, 1024]
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

        # this block output is cell current map
        self.score_block1 = nn.Sequential(

            nn.Conv2d(filters[0], self.n_classes, 5, padding=2),  # 5的卷积核大小
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
        model = RefinementModule().cuda()  # defined in the file of segmentation_refinement.models.psp.pspnet
        output = process_high_res_im(model, input, h5)
        output = (output[0, 0].cpu().numpy() * 255).astype('uint8')

        return output

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size()).cuda(0)



##############################################
#The module of R2U-Net comes from the paper ""
##############################################
"""  
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):  #输入输出通道一样,不改变图像大小
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        #print("##########The x is ########",np.shape(x))
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x1    #源码为x+x1

class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        #print("********The shape of x is *********",np.shape(x))  #torch.Size([4, 3, 96, 96])
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
"""

"""
#############################
#The code comes from the program of https://github.com/ZiyuanMa/U-Net/blob/master/model.py
#############################
class RC_block(nn.Module):
    def __init__(self, channel, t=2):
        super().__init__()
        self.t = t

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r_x = self.conv(x)

        for _ in range(self.t):
            r_x = self.conv(x + r_x)

        return r_x

class RRC_block(nn.Module):
    def __init__(self, channel, t=2):
        super().__init__()

        self.RC_net = nn.Sequential(
            RC_block(channel, t=t),
            RC_block(channel, t=t),
        )

    def forward(self, x):
        res_x = self.RC_net(x)

        return x + res_x

class R2UNet(nn.Module):
    def __init__(self):
        super(R2UNet,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            RRC_block(64),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            RRC_block(128),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            RRC_block(256),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            RRC_block(512),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            RRC_block(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            RRC_block(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            RRC_block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            RRC_block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            RRC_block(64),
            nn.Conv2d(64, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)

        x = self.up_conv1(torch.cat((x, x4), dim=1))
        x = self.up_conv2(torch.cat((x, x3), dim=1))
        x = self.up_conv3(torch.cat((x, x2), dim=1))
        x = self.final_conv(torch.cat((x, x1), dim=1))

        x = self.sigmoid(x)

        return x
"""