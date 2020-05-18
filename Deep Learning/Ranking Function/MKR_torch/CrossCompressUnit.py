import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader,Sampler,Dataset,TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from utils import load_data,load_rating,dataset_split,load_kg,get_user_record

class CrossCompressUnit(nn.Module):
    def __init__(self,dim,name=None):
        super(CrossCompressUnit,self).__init__()
        self.weight_vv = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(dim,1)), requires_grad=True)
        self.weight_ev = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(dim,1)), requires_grad=True)
        self.weight_ve = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(dim,1)), requires_grad=True)
        self.weight_ee = nn.Parameter(nn.init.kaiming_uniform_(torch.FloatTensor(dim,1)), requires_grad=True)
        
        self.bias_v = nn.Parameter(torch.FloatTensor(dim),requires_grad=True)
        self.bias_e = nn.Parameter(torch.FloatTensor(dim),requires_grad=True)
        
        self.dim = dim
    def forward(self,inputs):
        
        v,e = inputs
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = v.unsqueeze(2)
        e = e.unsqueeze(1)
        
        # [batch_size, dim, dim]
        c_matrix = v*e
        c_matrix_transpose = c_matrix.permute([0,2,1])
        
        #[batch_size * dim, dim]
        c_matrix = c_matrix.reshape([-1,self.dim])
        c_matrix_transpose = c_matrix.reshape([-1,self.dim])
        
        v_output = c_matrix.mm(self.weight_vv) + c_matrix_transpose.mm(self.weight_ev)
        v_output = v_output.reshape([-1,self.dim])
        v_output = v_output+self.bias_v
        
        e_output = c_matrix.mm(self.weight_ve) + c_matrix_transpose.mm(self.weight_ee)
        e_output = e_output.reshape([-1,self.dim])
        e_output = e_output+self.bias_e
        
        return v_output,e_output

