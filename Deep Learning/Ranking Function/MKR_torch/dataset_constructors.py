import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader,Sampler,Dataset,TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from utils import load_data,load_rating,dataset_split,load_kg,get_user_record

class MKR_RS_Dataset(Dataset):
    def __init__(self,train_dt):
        super(MKR_RS_Dataset,self).__init__()
        self.user_id = train_dt[:,0]
        self.item_id = train_dt[:,1]
        self.y = train_dt[:,2]
        self.n = train_dt.shape[0]
        
    def __getitem__(self,index):
        return [torch.tensor(self.user_id[index],dtype=torch.long),\
                torch.tensor(self.item_id[index],dtype=torch.long),\
                torch.tensor(self.y[index],dtype=torch.long)]

    def __len__(self):
        return self.n
    

class MKR_KGE_Dataset(Dataset):
    def __init__(self,kg_dt):
        super(MKR_KGE_Dataset,self).__init__()
        self.head_id = kg_dt[:,0]
        self.relation_id = kg_dt[:,1]
        self.tail_id = kg_dt[:,2]
        self.n = kg_dt.shape[0]
    
    def __getitem__(self,index):
        return [torch.tensor(self.head_id[index],dtype=torch.long),\
                torch.tensor(self.relation_id[index],dtype=torch.long),\
                torch.tensor(self.tail_id[index],dtype=torch.long)
               ]
    def __len__(self):
        return self.n
