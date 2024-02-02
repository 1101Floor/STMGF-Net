import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim




class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.residual = residual
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,(kernel_size[0], 1),(stride, 1),padding,),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.tcn(x) + self.residual(x)
        x = self.prelu(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else input_size
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        return y

class Agg(nn.Module):
    def __init__(self, ch_in=8, reduction=1):##reduction an be set
        super(Agg, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        beta = torch.softmax(y, dim=0)
        return (x * beta).sum(0),beta

class Fusion(nn.Module):
    def __init__(self, ch_in=8, reduction=1):
        super(Fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        beta = torch.softmax(y, dim=0)
        return (x * beta).sum(0),beta


class SE(nn.Module):
    def __init__(self, ch_in=8, reduction=1):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        beta = torch.softmax(y, dim=0)
        return (x * beta).sum(0),beta

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,input_feat=2,output_feat=5,
                 seq_len=20,pred_seq_len=10,kernel_size=3,num_channels=[10,10,10]):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
                
        self.st_gcns_dis = nn.ModuleList()
        self.st_gcns_dis.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        self.st_gcns_dpca = nn.ModuleList()
        self.st_gcns_dpca.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        self.st_gcns_tpca = nn.ModuleList()
        self.st_gcns_tpca.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        self.st_gcns_vs = nn.ModuleList()
        self.st_gcns_vs.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        self.st_gcns_sim = nn.ModuleList()
        self.st_gcns_sim.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        self.agg = Agg(ch_in=output_feat, reduction=1)
        self.Dy_fusion = Fusion(ch_in=output_feat, reduction=1)
        self.sfatt = SE(ch_in=output_feat, reduction=1)

        self.TCN=TCN(seq_len, pred_seq_len,num_channels, (kernel_size, seq_len))
        self.TCN_ouput = nn.Conv2d(num_channels[-1],pred_seq_len,3,padding=1)

    def forward(self,v,a_dis,a_dpca,a_tpca,a_s,a_sim):
        v_copy = v

        for k in range(self.n_stgcnn):
            v_dis,a_dis = self.st_gcns_dis[k](v,a_dis)

        for k in range(self.n_stgcnn):
            v_dpca,a_dpca = self.st_gcns_dpca[k](v,a_dpca)

        for k in range(self.n_stgcnn):
            v_tpca,a_tpca = self.st_gcns_tpca[k](v,a_tpca)

        for k in range(self.n_stgcnn):
            v_s,a_s = self.st_gcns_vs[k](v,a_s)

        for k in range(self.n_stgcnn):
            v_sim,a_sim = self.st_gcns_vs[k](v,a_sim)

        v_ad = torch.stack([v_dis, v_dpca, v_tpca], dim=1)
        vec = torch.zeros(v_ad.shape[0],v_ad.shape[2],v_ad.shape[3],v_ad.shape[4])
        for i in range(v_ad.shape[0]):

            v_att, att = self.agg(v_ad[i])
            vec[i]=v_att

        v_add = torch.stack([vec,v_s,v_sim], dim=1)
        vec1 = torch.zeros(v_add.shape[0], v_add.shape[2], v_add.shape[3], v_add.shape[4])
        for i in range(v_add.shape[0]):
            v_att1, att1 = self.Dy_fusion(v_add[i])
            vec1[i] = v_att1

        v_sfatt, sfatt = self.sfatt(vec1)
        v_end = vec1.view(v_dis.shape[0],v_dis.shape[2],v_dis.shape[1],v_dis.shape[3])#1,8,5,7

        v_end = self.TCN(v_end)
        v_end = self.TCN_ouput(v_end)

        v_end = v_end.view(v_end.shape[0], v_end.shape[2], v_end.shape[1], v_end.shape[3])
        v_end = sfatt * v_end

        return v_end,(a_dis,a_dpca,a_tpca,a_s,a_sim)