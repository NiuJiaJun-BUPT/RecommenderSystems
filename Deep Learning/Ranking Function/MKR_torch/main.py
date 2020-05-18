import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader,Sampler,Dataset,TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from utils import load_data,load_rating,dataset_split,load_kg,get_user_record,eval_model
from models import CrossCompressUnit,MKR_torch
from dataset_constructors import MKR_RS_Dataset,MKR_KGE_Dataset
parser = argparse.ArgumentParser()

# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=0.02, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=0.01, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=3, help='training interval of KGE task')

args = parser.parse_args()

data = load_data(args)

show_loss = False
show_topk = False

#分解数据
n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
train_data, eval_data, test_data = data[4], data[5], data[6]
kg = data[7]

user_num = 100
k_list = [1, 2, 5, 10, 20, 50, 100]
#获取用户交互记录
train_record = get_user_record(train_data, True)
test_record = get_user_record(test_data, False)
#全部用户列表
user_list = list(set(train_record.keys()) & set(test_record.keys()))

#取出100个用户
if len(user_list) > user_num:
    user_list = np.random.choice(user_list, size=user_num, replace=False)
#取出全部item
item_set = set(list(range(n_item)))

mkdr_rs_ds = MKR_RS_Dataset(train_data)
mkdr_rs_dl = DataLoader(mkdr_rs_ds,batch_size=args.batch_size,shuffle=True)

mkdr_kge_ds = MKR_KGE_Dataset(kg)
mkdr_kge_dl = DataLoader(mkdr_kge_ds,batch_size=args.batch_size,shuffle=True)

model = MKR_torch(args,n_user,n_item,n_entity,n_relation)
print(model)
optimizer_rs=torch.optim.Adam(model.parameters(),lr=args.lr_rs,weight_decay=1e-6)
optimizer_kge=torch.optim.Adam(model.parameters(),lr=args.lr_kge,weight_decay=1e-6)

for step in range(100): #args.n_epochs
    start = 0
    #训练RS
    for i, data in enumerate(mkdr_rs_dl,0):
        
        u_id,i_id,label = data
        optimizer_rs.zero_grad()

        scores = model.forward_rs(u_id,i_id,i_id,use_inner_product=False) #[batch_size], 0-1的概率
        rs_loss_func = torch.nn.BCEWithLogitsLoss()
        label = torch.tensor(label,dtype=torch.float32)

        loss_rs = rs_loss_func(scores,label)
        loss_rs.backward()
        optimizer_rs.step()
        if i%10 == 0:
            print(loss_rs) 
        
    #训练KGE
    for i, data in enumerate(mkdr_kge_dl,0):
        head_id,relation_id,tail_id = data
        optimizer_kge.zero_grad()
        
        kge_loss,kge_rmse = model.forward_kge(head_id,head_id,relation_id,tail_id)
    
        kge_loss.backward()
        optimizer_kge.step()
    print(kge_loss,kge_rmse)  

    #CTR评估
    train_auc, train_acc = eval_model(MKR_RS_Dataset(train_data),model,args)
    print("train: ",train_auc,train_acc)
    eval_auc,eval_acc = eval_model(MKR_RS_Dataset(eval_data),model,args)
    print("eval: ",eval_auc,eval_acc)
    test_auc,test_acc = eval_model(MKR_RS_Dataset(test_data),model,args)
    print("test: ",test_auc,test_acc)