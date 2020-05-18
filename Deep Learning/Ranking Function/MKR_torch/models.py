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
    
class MKR_torch(nn.Module):
    def __init__(self,args,n_users, n_items, n_entities, n_relations):
        super(MKR_torch, self).__init__()
        self.args = args
        self._parse_args(n_users, n_items, n_entities, n_relations)
        self._build_model(args)
#         self._build_loss(args)
#         self._build_train(args)
        
    def _parse_args(self, n_users, n_items, n_entities, n_relations):
        self.n_user = n_users
        self.n_item = n_items
        self.n_entity = n_entities
        self.n_relation = n_relations

        # for computing l2 loss
        self.vars_rs = []
        self.vars_kge = []
    
    def forward_rs(self,user_indices,item_indices,head_indices,use_inner_product=True):
        #batch_size,dim -> [4096,8]
        self.user_embeddings = self.user_emb_matrix(user_indices)
        self.item_embeddings = self.item_emb_matrix(item_indices)
        self.head_embeddings = self.entity_emb_matrix(head_indices)

#         for i in range(args.L):
#             self.user_embeddings = self.user_mlp[i](self.user_embeddings)
#             self.item_embeddings,self.head_embeddings = self.cc_unit[i]([self.item_embeddings,self.head_embeddings])
        
        if use_inner_product:
            self.scores = torch.sum(self.user_embeddings*self.item_embeddings,axis=1)
        else:
            self.user_item_concat = torch.cat([self.user_embeddings,self.item_embeddings],axis=1)
            for i in range(self.args.H-1):
                self.user_item_concat=self.rs_mlps[i](self.user_item_concat)
            self.scores = self.rs_pred_mlp(self.user_item_concat).squeeze()
#         self.scores = torch.sigmoid(self.scores)
        return self.scores
    
    def forward_kge(self,item_indices,head_indices,relation_indices,tail_indices):
        self.item_embeddings = self.item_emb_matrix(item_indices)
        self.head_embeddings = self.entity_emb_matrix(head_indices)
        self.relation_embeddings = self.relation_emb_matrix(relation_indices)
        self.tail_embeddings = self.entity_emb_matrix(tail_indices)
        
        for i in range(self.args.L):
            self.relation_embeddings = self.rel_mlp[i](self.relation_embeddings)
            
        self.head_relation_concat = torch.cat([self.head_embeddings,self.relation_embeddings],axis=1)
        for i in range(self.args.H-1):
            self.head_realation_concat = self.kge_mlps[i](self.head_relation_concat)
            
        self.tail_pred = self.kge_pred_mlp(self.head_relation_concat)
        self.tail_pred = torch.sigmoid(self.tail_pred)
        #在这里直接计算损失
        self.scores_kge = torch.sum(torch.sigmoid(torch.sum(self.tail_embeddings*self.tail_pred,axis=1)))
        self.rmse=torch.mean(torch.sqrt(torch.sum((self.tail_embeddings-self.tail_pred)**2,axis=1)/self.args.dim))
        self.base_loss_kge = -self.scores_kge
        
        #加入l2损失
        self.l2_loss_kge = torch.sum(self.head_embeddings**2)/2 +torch.sum(self.tail_embeddings**2)/2
        self.l2_loss_kge = self.l2_loss_kge*self.args.l2_weight
        self.loss_kge = self.base_loss_kge + self.l2_loss_kge
#         print("loss_kge",self.loss_kge,"base",self.base_loss_kge,"l2",self.l2_loss_kge)
        return self.loss_kge,self.rmse
        
    def _build_model(self,args):
        self._build_low_layers(args)
        self._build_high_layers(args)
        
    def _build_low_layers(self,args):
        #Embedding构建
        self.user_emb_matrix = nn.Embedding(self.n_user,args.dim)
        self.item_emb_matrix = nn.Embedding(self.n_item,args.dim)
        self.entity_emb_matrix = nn.Embedding(self.n_entity,args.dim)
        self.relation_emb_matrix=  nn.Embedding(self.n_relation,args.dim)
        
        self.user_mlp=nn.ModuleList()
        self.rel_mlp=nn.ModuleList()
        self.cc_unit=nn.ModuleList()
        
        
        for _ in range(args.L):
            self.user_mlp.append(nn.Linear(args.dim,args.dim))
            self.rel_mlp.append(nn.Linear(args.dim,args.dim))
            self.cc_unit.append(CrossCompressUnit(args.dim))
            
    def _build_high_layers(self,args):
        #RS
        #拼接方法时过的mlp
        self.rs_mlps = nn.ModuleList()
        for _ in range(args.H-1):
            rs_mlp = nn.Linear(args.dim*2, args.dim*2)
            #[batch_size, dim*2]
            self.rs_mlps.append(rs_mlp)
        
        self.rs_pred_mlp = nn.Linear(args.dim*2,1)
        
        #KGE
        self.kge_mlps = nn.ModuleList()
        for _ in range(args.H-1):
            kge_mlp = nn.Linear(args.dim*2,args.dim*2)
            self.kge_mlps.append(kge_mlp)
            
        self.kge_pred_mlp = nn.Linear(args.dim*2,args.dim)
            
        