import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader,Sampler,Dataset,TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import os
# from dataset_constructors import MKR_RS_Dataset,MKR_KGE_Dataset

base_dir = "./"
def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, kg


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = base_dir+'data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = base_dir+'data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    return n_entity, n_relation, kg
def get_user_record(data, is_train):
    #建立每一个user交互过的item字典
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
def eval_model(mkdr_rs_ds,model,args):
    model.eval()
#     mkdr_rs_ds = MKR_RS_Dataset(data)
    mkdr_rs_dl = DataLoader(mkdr_rs_ds,batch_size=args.batch_size,shuffle=False)
    auc_score_list = []
    auc_label_list = []
    acc_list = []
    for i, data in enumerate(mkdr_rs_dl,0):
        u_id,i_id,label = data
        scores = model.forward_rs(u_id,i_id,i_id,False)
        scores_norm = torch.sigmoid(scores).cpu().detach().numpy()
        label = label.numpy()

        predictions = [1 if i >= 0.5 else 0 for i in scores_norm]
        auc_score_list = auc_score_list + list(scores_norm)
        auc_label_list = auc_label_list + list(label)
        
        predictions = np.array(predictions)
        
#         auc = roc_auc_score(label,predictions)
#         auc_list.append(auc)
        
        acc = np.mean(np.equal(predictions,label))
        acc_list.append(acc)
    auc = roc_auc_score(auc_label_list,auc_score_list)
    return auc,np.mean(acc)
