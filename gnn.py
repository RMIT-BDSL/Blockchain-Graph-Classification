import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as tg
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import time
import yaml

from logger import Logger
from torch_geometric.utils import to_undirected
from utils import DGraphFin, Structure, Missvalues, Background
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from model import GAT, GCN, SAGE

def train(model, data, train_idx, optimizer, weight=None, no_conv=False, is_rgcn=False):
    model.train()
    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        if is_rgcn:
            out = model(data.x, data.edge_index, data.edge_type)[train_idx]
        else:
            out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx], weight=weight)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False, is_rgcn=False):
    model.eval()
    if no_conv:
        out = model(data.x)
    else:
        if is_rgcn:
            out = model(data.x, data.edge_index, data.edge_type)
        else:
            out = model(data.x, data.edge_index)
    y_pred = out.exp()  # Convert log-probabilities to probabilities.
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])
    return eval_results, losses, y_pred

def main():
    # Parse all arguments (including the config file) in one place.
    parser = argparse.ArgumentParser(description="Run experiments with YAML configuration and model parameters")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp')  # Options: 'mlp', 'gcn', 'sage'
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--MV_trick', type=str, default='null')
    parser.add_argument('--BN_trick', type=str, default='null')
    parser.add_argument('--BN_ratio', type=float, default=0.1)
    parser.add_argument('--Structure', type=str, default='original')
    args = parser.parse_args()
    print(args)
    
    # Load the YAML config.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    eval_metric = config.get('eval_metric', 'auc')
    mlp_params = config.get('mlp_parameters', {})
    gcn_params = config.get('gcn_parameters', {})
    sage_params = config.get('sage_parameters', {})
    
    # Decide if the model uses convolution layers.
    no_conv = (args.model == 'mlp')
    
    device_str = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    # Load dataset.
    dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
    data = dataset[0]
    
    # Convert sparse representation to edge_index.
    data.edge_index = data.adj_t
    data.adj_t = torch.cat([
        data.edge_index.coo()[0].view(1, -1),
        data.edge_index.coo()[1].view(1, -1)
    ], dim=0)
    data.edge_index = data.adj_t
    
    # Process structure.
    structure = Structure(args.Structure)
    data = structure.process(data)
    data.adj_t = data.edge_index
    data.adj_t = tg.utils.to_undirected(data.adj_t)
    data.edge_index = data.adj_t
    
    # Normalize features for DGraphFin.
    if args.dataset == 'DGraphFin':
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x
    
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    
    print(data)
    
    # Process missing values.
    missvalues = Missvalues(args.MV_trick)
    data = missvalues.process(data)
    
    data.edge_index = data.adj_t
    # Process background (e.g. applying BN trick).
    BN = Background(args.BN_trick)
    data = BN.process(data, args.BN_ratio)
    
    # Setup train/valid/test splits.
    split_idx = {
        'train': data.train_mask,
        'valid': data.valid_mask,
        'test': data.test_mask
    }
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        fold = args.fold
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
    
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)
    
    # Initialize model.
    is_rgcn = False
    nlabels = 2 if args.dataset == 'DGraphFin' else dataset.num_classes
    if args.model == 'sage':
        para_dict = sage_params
        model_para = sage_params.copy()
        model_para.pop('lr', None)
        model_para.pop('l2', None)
        model = SAGE(
            in_channels=data.x.size(-1),
            out_channels=nlabels,
            **model_para
        )
    elif args.model == 'gcn':
        para_dict = gcn_params
        model = GCN(
            in_channels=data.x.size(-1),
            hidden_channels=gcn_params['hidden_channels'],
            out_channels=nlabels,
            num_layers=gcn_params['num_layers'],
            dropout=gcn_params['dropout'],
            batchnorm=gcn_params['batchnorm']
        )
    elif args.model == 'mlp':
        para_dict = mlp_params
        # Define a simple MLP.
        class MLP(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm):
                super(MLP, self).__init__()
                layers = []
                if num_layers == 1:
                    layers.append(nn.Linear(in_channels, out_channels))
                else:
                    layers.append(nn.Linear(in_channels, hidden_channels))
                    if batchnorm:
                        layers.append(nn.BatchNorm1d(hidden_channels))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    for _ in range(num_layers - 2):
                        layers.append(nn.Linear(hidden_channels, hidden_channels))
                        if batchnorm:
                            layers.append(nn.BatchNorm1d(hidden_channels))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                    layers.append(nn.Linear(hidden_channels, out_channels))
                self.network = nn.Sequential(*layers)
            
            def reset_parameters(self):
                for m in self.network:
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            
            def forward(self, x, edge_index=None):
                return self.network(x)
        
        model = MLP(
            in_channels=data.x.size(-1),
            hidden_channels=mlp_params['hidden_channels'],
            out_channels=nlabels,
            num_layers=mlp_params['num_layers'],
            dropout=mlp_params['dropout'],
            batchnorm=mlp_params['batchnorm']
        )
    else:
        raise ValueError("Unknown model specified!")
    
    model = model.to(device)
    print(f'Model {args.model} initialized')
    
    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)
    weight = torch.tensor([1, 50]).to(device).float()
   
    for run in range(args.runs):
        import gc
        gc.collect()
        print("Number of parameters:", sum(p.numel() for p in model.parameters()))
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        min_valid_loss = float('inf')
        best_out = None
        
        time_ls = []
        for epoch in range(1, args.epochs + 1):
            starttime = time.time()
            loss = train(model, data, train_idx, optimizer, weight, no_conv, is_rgcn)
            endtime = time.time()
            time_ls.append(endtime - starttime)
            
            eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv, is_rgcn)
            train_auc = eval_results['train'].get('auc', 0)
            valid_auc = eval_results['valid'].get('auc', 0)
            test_auc  = eval_results['test'].get('auc', 0)
            train_loss = losses['train']
            valid_loss = losses['valid']
            
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()
            
            if epoch % args.log_steps == 0:
                train_ap = eval_results['train'].get('ap', 0)
                valid_ap = eval_results['valid'].get('ap', 0)
                test_ap  = eval_results['test'].get('ap', 0)
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train AUC: {train_auc:.3f} Train AP: {train_ap:.3f}, '
                      f'Valid AUC: {valid_auc:.3f} Valid AP: {valid_ap:.3f}, '
                      f'Test AUC: {test_auc:.3f} Test AP: {test_ap:.3f}, '
                      f'Avg Train Time: {np.mean(time_ls):.3f}s')
                time_ls = []
            logger.add_result(run, [train_auc, valid_auc, test_auc])
        
        logger.print_statistics(run)
    
    final_results = logger.print_statistics()
    print('final_results:', final_results)
    para_dict.update(final_results)
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir+'/results.csv')

if __name__ == "__main__":
    main()
