import os
import sys
import gc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import math
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from tqdm.auto import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, subgraph
from yaml import safe_load
from model.GAT import *
from model.GCN import *

model_dict = {
    'GAT': GAT,
    'GATv2': GATv2,
    'GCN': GCN
}

# -------------------- Setup Logger --------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# -------------------- Load Configs --------------------
model_config = safe_load(open('config/model.yaml', 'r'))
training_config = safe_load(open('config/training.yaml', 'r'))

# -------------------- Load Dataset --------------------
logger.info("Loading dataset...")
with torch.serialization.safe_globals([Data]):
    data = torch.load(training_config['data']['data_path'], weights_only=training_config['data']['weights_only'])
logger.info(f"Nodes: {data.num_nodes}, Features: {data.num_node_features}")

# -------------------- Device Setup --------------------
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# -------------------- Model Setup --------------------
model = model_dict[model_config['model']['type']](
    in_channels=data.num_node_features,
    **model_config['model']['params']
)
model = model.to(device)
logger.info(f"Model: \n{model}")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# -------------------- Optimizer Setup --------------------
optimizer_config = training_config['optimizer']
logger.info(f"Optimizer: {optimizer_config['type']}")

if optimizer_config['type'].lower() == 'adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay']
    )
elif optimizer_config['type'].lower() == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay'],
        momentum=optimizer_config['momentum']
    )
else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    
# -------------------- Loss Function Setup --------------------
criterion = torch.nn.NLLLoss()
logger.info(f"Loss function: {criterion.__class__.__name__}")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', factor=0.5, patience=10, verbose=True
)
logger.info(f"Scheduler: {scheduler.__class__.__name__}")

# -------------------- Pre-training Setup --------------------
if model_config['pretrained']:
    checkpoint_path = model_config['checkpoint']
    logger.info(f"Finetuning from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Pretrained weights loaded successfully.")

# -------------------- Training Setup ---------------------
for param in model.parameters():
    param.requires_grad = True
save_path = model_config['checkpoint'].replace('.pt', f"_{model.__class__.__name__}.pt")

# -------------------- Training Loop --------------------
best_val_acc = 0
train_losses = []
val_losses = []
patience = training_config['training']['patience']
early_stopping_counter = 0

logger.info("Starting training...")
for epoch in tqdm(range(1, training_config['training']['epochs'] + 1), desc="Training"):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    train_loss = criterion(out, data.y)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_labels = data.y
        val_loss = criterion(val_out, val_labels)
        val_losses.append(val_loss.item())

        val_pred = val_out.argmax(dim=1)
        val_acc = (val_pred == val_labels).float().mean().item()
    
    # Step scheduler
    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stopping_counter = 0
        torch.save({'model_state_dict': model.state_dict()}, save_path)
    else:
        early_stopping_counter += 1
        
    if early_stopping_counter >= patience:
        logger.info(f"Early stopping triggered at epoch {epoch:03d}. Best Val Acc: {best_val_acc:.4f}")
        break
    tqdm.write(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

#--------------------- Plotting Loss and Accuracy --------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png")
logger.info("Saved loss curve to loss_curve.png")