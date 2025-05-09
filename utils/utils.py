import torch
import numpy as np
import os
from datetime import datetime
import shutil


def prepare_folder(name, model_name):
    model_dir = f'./model_results/{name}/{model_name}'
   
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir

def prepare_tune_folder(name, model_name):
    str_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    tune_model_dir = f'./tune_results/{name}/{model_name}/{str_time}/'
   
    if os.path.exists(tune_model_dir):
        print(f'rm tune_model_dir {tune_model_dir}')
        shutil.rmtree(tune_model_dir)
    os.makedirs(tune_model_dir)
    print(f'make tune_model_dir {tune_model_dir}')
    return tune_model_dir

def save_preds_and_params(parameters, preds, model, file):
    save_dict = {'parameters':parameters, 'preds': preds, 'params': model.state_dict()
           , 'nparams': sum(p.numel() for p in model.parameters())}
    torch.save(save_dict, file)
    return 
    
    

class Structure:
    def __init__(self,trick):
        if trick not in ['original', 'knn', 'random']:
            raise ValueError('trick should be original, knn, or random')
        self.trick = trick
    def process(self,data):
        if self.trick == 'original':
            return data
        if self.trick == 'knn':
            return self._KNNGraph(data)
        if self.trick == 'random':
            return self._RandomGraph(data)
    def _KNNGraph(self,data,num=4300999):
        num=4300999
        node_num = data.x.shape[0]
        L = (torch.rand(num*50)*node_num).long().view(-1,1)
        R = (torch.rand(num*50)*node_num).long().view(-1,1)
        flag = (L.view(-1)!=R.view(-1))
        L = L[flag,:]
        R = R[flag,:]
        edge_index = torch.cat((L,R),dim=1).T
        x = data.x/data.x.norm(dim=1).view(-1,1)
        L = x[L.view(-1)]
        R = x[R.view(-1)]
        score = (L*R).sum(dim=1)
        score_a= score[score>0.9]
        edge_index = edge_index[:,score>0.9]
        score = score_a.numpy()
        index = np.argsort(-score)
        index = torch.tensor(index).long()
        edge_index = edge_index[:,index[:4300999]]
        data.edge_index = edge_index
        return data
    def _RandomGraph(self,data,num=4300999):
        node_num = data.x.shape[0]
        L = (torch.rand(num*2)*node_num).long().view(-1,1)
        R = (torch.rand(num*2)*node_num).long().view(-1,1)
        flag = (L.view(-1)!=R.view(-1))
        L = L[flag,:]
        R = R[flag,:]
        L = L[:num,:]
        R = R[:num,:]
        edge_index = torch.cat((L,R),dim=1).T
        data.edge_index = edge_index
        return data
