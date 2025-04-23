import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import time
import gc
import yaml

from logger import Logger
from torch_geometric.data import Data
from utils import DGraphFin, Structure
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2  # Adjust model import as needed

# Evaluation metric
eval_metric = 'auc'

# Parameter dictionaries for each model
mlp_parameters = {
    'lr': 0.01,
    'num_layers': 2,
    'hidden_channels': 128,
    'dropout': 0.0,
    'batchnorm': False,
    'l2': 5e-7
}

gcn_parameters = {
    'lr': 0.01,
    'num_layers': 2,
    'hidden_channels': 128,
    'dropout': 0.0,
    'batchnorm': False,
    'l2': 5e-7
}

sage_parameters = {
    'lr': 0.01,
    'num_layers': 2,
    'hidden_channels': 128,
    'dropout': 0,
    'batchnorm': False,
    'l2': 5e-7
}


def train(model, data, train_idx, optimizer, no_conv=False):
    """Trains model on training indices."""
    model.train()
    optimizer.zero_grad()
    if no_conv:
        out = model(data.x[train_idx])
    else:
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    """Evaluates model on provided splits."""
    model.eval()
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
    y_pred = out.exp()  # Convert log-probabilities to probabilities.
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
    return eval_results, losses, y_pred


def main():
    parser = argparse.ArgumentParser(description='GNN models on DGraphFin')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='mlp',
                        help="Options: mlp, gcn, sage")
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    print(args)
    
    # Load YAML configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    # For DGraphFin we only need the gcn and sage parameters.
    gcn_params = config.get('gcn_parameters', gcn_parameters)
    sage_params = config.get('sage_parameters', sage_parameters)
    
    # Determine if a model without convolutions is used.
    no_conv = (args.model == 'mlp')
    
    # Set device.
    device_str = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    # ---------------------------
    # Load and preprocess DGraphFin dataset.
    # ---------------------------
    print("Loading DGraphFin dataset...")
    dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
    data = dataset[0]
    
    # Ensure symmetric adjacency.
    data.adj_t = data.adj_t.to_symmetric()
    
    # Normalize features.
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
    
    # If labels have extra dimension, squeeze it.
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)
    
    # Create split indices from the precomputed masks.
    split_idx = {
        'train': data.train_mask,
        'valid': data.valid_mask,
        'test': data.test_mask
    }
    
    # Handle k-fold splits if available.
    if split_idx['train'].dim() > 1 and split_idx['train'].shape[1] > 1:
        print(f"There are {split_idx['train'].shape[1]} folds of splits")
        split_idx['train'] = split_idx['train'][:, args.fold]
        split_idx['valid'] = split_idx['valid'][:, args.fold]
        split_idx['test'] = split_idx['test'][:, args.fold]
    
    # Move data and masks to device.
    data = data.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    train_idx = split_idx['train']
    
    result_dir = prepare_folder(args.dataset, args.model)
    print('Result directory:', result_dir)
    
    # ---------------------------
    # Model initialization.
    # ---------------------------
    nlabels = dataset.num_classes
    if args.dataset in ['DGraphFin']:
        nlabels = 2  # Only two classes: normal and fraud.
    
    if args.model == 'mlp':
        # Remove lr and l2 from MLP's constructor arguments.
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
        para_dict = mlp_parameters
    elif args.model == 'gcn':
        model_para = gcn_params.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = GCN(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
        para_dict = gcn_params
    elif args.model == 'sage':
        model_para = sage_params.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = SAGE(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
        para_dict = sage_params
    else:
        raise ValueError("Unknown model specified!")
    
    print(f'Model {args.model} initialized with {sum(p.numel() for p in model.parameters())} parameters.')
    
    evaluator = Evaluator(eval_metric)
    logger = Logger(args.runs, args)
    
    # ---------------------------
    # Training loop.
    # ---------------------------
    for run in range(args.runs):
        gc.collect()
        print(f"Run {run + 1}, Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        min_valid_loss = float('inf')
        best_out = None

        for epoch in range(1, args.epochs + 1):
            loss = train(model, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test(model, data, split_idx, evaluator, no_conv)
            train_eval = eval_results['train']
            valid_eval = eval_results['valid']
            test_eval = eval_results['test']
            
            if losses['valid'] < min_valid_loss:
                min_valid_loss = losses['valid']
                best_out = out.cpu()
            
            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}%, '
                      f'Test: {100 * test_eval:.3f}%')
                # Optionally, print average training time, etc.
            logger.add_result(run, [train_eval, valid_eval, test_eval])
        
        logger.print_statistics(run)
    
    final_results = logger.print_statistics()
    print('Final results:', final_results)
    para_dict.update(final_results)
    pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir + '/results.csv')


if __name__ == "__main__":
    main()
