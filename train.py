import os
import sys
import yaml
import gc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import math
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from importlib import import_module
from tqdm.auto import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, subgraph
from yaml import safe_load
import model

# one-liner that pulls each symbol from the package into a dict
model_dict = { 
    name: getattr(model, name) for name in model.__all__
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

# -------------------- Data Preprocessing --------------------
torch.manual_seed(training_config['training']['seed'])
logger.info("Random seed set to: {}".format(training_config['training']['seed']))
logger.info("Preprocessing data...")
ratio = training_config['training']['ratio']
# [0.8, 0.1, 0.1] split
mask_known = (data.y != 2)
known_indices = torch.where(mask_known)[0]
perm = torch.randperm(len(known_indices))
train_size = int(len(perm) * ratio[0])
val_size = int(len(perm) * ratio[1])

train_indices = known_indices[perm[:train_size]]
valid_indices = known_indices[perm[train_size:train_size + val_size]]
test_indices = known_indices[perm[train_size + val_size:]]

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

logger.info(f"Using device: {device}")
data = data.to(device)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)

data.train_mask[train_indices] = True
data.val_mask[valid_indices] = True
data.test_mask[test_indices] = True
assert (data.train_mask & data.val_mask).sum().item() == 0
assert (data.train_mask & data.test_mask).sum().item() == 0
assert (data.val_mask & data.test_mask).sum().item() == 0

logger.info(f"Train: {data.train_mask.sum().item()}, Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}")

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
    checkpoint_path = f"{model.__class__.__name__}/{model_config['checkpoint']}"
    logger.info(f"Finetuning from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Pretrained weights loaded successfully.")

# -------------------- Training Setup ---------------------
for param in model.parameters():
    param.requires_grad = True
if not os.path.exists(f"{model_config['checkpoint']}/{model.__class__.__name__}"):
    os.makedirs(f"{model_config['checkpoint']}/{model.__class__.__name__}")
save_path = f"{model_config['checkpoint']}/{model.__class__.__name__}/{model.__class__.__name__}.pt"

# -------------------- Training Loop --------------------
best_val_acc = 0
train_losses = []
val_losses = []
val_accuracies = []
patience = training_config['training']['patience']
early_stopping_counter = 0

logger.info("Starting training...")
for epoch in tqdm(range(1, training_config['training']['epochs'] + 1), desc="Training"):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_losses.append(val_loss.item())

        pred = out[data.val_mask].argmax(dim=1)
        acc = (pred == data.y[data.val_mask]).float().mean().item()

    val_accuracies.append(acc)
    scheduler.step(acc)

    if acc > best_val_acc:
        best_val_acc = acc
        early_stopping_counter = 0
        torch.save({'model_state_dict': model.state_dict()}, save_path)
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        logger.info(f"Early stopping at epoch {epoch:03d}. Best Val Acc: {best_val_acc:.4f}")
        break
    tqdm.write(f"[Epoch {epoch:03d}] Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}")
    gc.collect()
    torch.cuda.empty_cache()
    
logger.info(f"Training completed. Best Val Acc: {best_val_acc:.4f}")

# -------------------- Testing Loop --------------------
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_pred = out[data.test_mask].argmax(dim=1)
    test_acc = (test_pred == data.y[data.test_mask]).float().mean().item()
logger.info(f"Test Accuracy: {test_acc:.4f}")

# --------------------- Plotting Loss --------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
if not os.path.exists(f"results/{model.__class__.__name__}"):
    os.makedirs(f"results/{model.__class__.__name__}")
plt.savefig(f"results/{model.__class__.__name__}/loss_curve.png")
logger.info("Saved loss curve to loss_curve.png")

# -------------------- Plotting Accuracy --------------------
plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/{model.__class__.__name__}/accuracy_curve.png")
logger.info("Saved accuracy curve to accuracy_curve.png")

# -------------------- Save Configs --------------------
with open(f"{model_config['checkpoint']}/{model.__class__.__name__}/model.yaml", 'w') as f:
    yaml.safe_dump(model_config, f)
with open(f"{model_config['checkpoint']}/{model.__class__.__name__}/training.yaml", 'w') as f:
    yaml.safe_dump(training_config, f)
with open(f"results/{model.__class__.__name__}/test_accuracy.txt", 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.4f}")
    
logger.info("Saved model and training configurations.")
logger.info("Training and evaluation completed successfully.")
# -------------------- End of Script --------------------