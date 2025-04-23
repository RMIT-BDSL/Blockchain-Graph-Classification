import torch
import numpy as np

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
