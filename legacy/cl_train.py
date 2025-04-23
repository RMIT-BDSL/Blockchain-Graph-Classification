import os
import sys
import gc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import math
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, subgraph
from model.GCN import GCN

# -----------------------------------------------------------------------------
# 1. Load & inspect Elliptic dataset
# -----------------------------------------------------------------------------
edgelists = pd.read_csv('data/elliptic/elliptic_txs_edgelist.csv')
features  = pd.read_csv('data/elliptic/elliptic_txs_features.csv')
classes   = pd.read_csv('data/elliptic/elliptic_txs_classes.csv')

features.columns = ['txId'] + [f"V{i+1}" for i in range(features.shape[1]-1)]
print("Nodes:", len(features), "Edges:", len(edgelists))

classes['class_mapped'] = classes['class'].replace({'1':'illicit','2':'licit','3':'unknown'})
print("\nClass distribution:")
print(classes['class_mapped'].value_counts(normalize=True).mul(100).round(2))

# -----------------------------------------------------------------------------
# 2. Load preprocessed PyG Data
# -----------------------------------------------------------------------------
data_path = 'data/elliptic/elliptic_data.pt'
assert os.path.exists(data_path), f"{data_path} not found"
with torch.serialization.safe_globals([Data]):
    data = torch.load('data/elliptic/elliptic_data.pt', weights_only=False)
print(f"Loaded Data → Nodes: {data.num_nodes}, Features: {data.num_node_features}")

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# -----------------------------------------------------------------------------
# 3. Contrastive pre-training setup
# -----------------------------------------------------------------------------
# Hyperparameters
HIDDEN_DIM = 64
EMB_DIM    = 32
NUM_LAYERS = 3
DROPOUT    = 0.5
BATCH_SIZE = 1024
BATCHNORM  = True
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 100

# Contrastive GCN encoder: same layers as GCN, but out_channels=EMB_DIM
class ContrastiveGCN(torch.nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.encoder = GCN(
            in_channels   = in_feats,
            hidden_channels = HIDDEN_DIM,
            out_channels    = EMB_DIM,
            num_layers    = NUM_LAYERS,
            dropout       = DROPOUT,
            batchnorm     = BATCHNORM
        )
    def forward(self, x, edge_index):
        # returns embeddings [N, EMB_DIM]
        # Note: GCN.forward applies log_softmax; for contrastive we want raw pre-softmax,
        # so remove the final log_softmax. We can hack by calling convs manually:
        for i, conv in enumerate(self.encoder.convs[:-1]):
            x = conv(x, edge_index)
            if BATCHNORM:
                x = self.encoder.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=DROPOUT, training=self.training)
        # final layer without log_softmax:
        x = self.encoder.convs[-1](x, edge_index)
        return x

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2*N, device=sim.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N)
    ], dim=0)
    num = torch.exp(pos)
    den = torch.exp(sim).sum(dim=1)
    return -torch.log(num / den).mean()

# -----------------------------------------------------------------------------
# 4. Run Contrastive Pre-training
# -----------------------------------------------------------------------------
pretrain_model = ContrastiveGCN(data.num_node_features).to(device)
pretrain_opt   = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)

print("\n🔬 Contrastive pre-training:")
pretrain_model.train()
num_nodes = data.num_nodes

for epoch in range(1, PRETRAIN_EPOCHS+1):
    perm = torch.randperm(num_nodes, device=device)
    total_loss = 0.0
    steps = math.ceil(num_nodes / BATCH_SIZE)

    for i in range(0, num_nodes, BATCH_SIZE):
        batch_idx = perm[i:i + BATCH_SIZE]

        # induce a subgraph on these nodes:
        edge_idx_sub, _, _ = subgraph(batch_idx, data.edge_index,
                                      relabel_nodes=True,
                                      num_nodes=num_nodes)
        x_sub = data.x[batch_idx]

        # two views on the subgraph
        # view 1: drop edges
        e1, _ = dropout_edge(edge_idx_sub, p=0.2)
        # view 2: add feature noise
        x2 = x_sub + 0.1 * torch.randn_like(x_sub)

        # forward through encoder
        z1 = pretrain_model(x_sub,      e1)
        z2 = pretrain_model(x2,         edge_idx_sub)

        loss = nt_xent_loss(z1, z2)
        total_loss += loss.item()

        pretrain_opt.zero_grad()
        loss.backward()
        pretrain_opt.step()

    avg = total_loss / steps
    if epoch % 5 == 0:
        print(f"[Pretrain] Epoch {epoch:02d}/{PRETRAIN_EPOCHS}  Avg Loss: {avg:.4f}")

# -----------------------------------------------------------------------------
# 5. Fine-tune for 3-class classification
# -----------------------------------------------------------------------------
classifier = GCN(
    in_channels    = data.num_node_features,
    hidden_channels = HIDDEN_DIM,
    out_channels     = 3,           # 3 classes: illicit, licit, unknown
    num_layers     = NUM_LAYERS,
    dropout        = DROPOUT,
    batchnorm      = BATCHNORM
).to(device)

# copy pretrained GCN conv weights (all but final layer)
for i in range(NUM_LAYERS-1):
    classifier.convs[i].load_state_dict(pretrain_model.encoder.convs[i].state_dict())
    if BATCHNORM:
        classifier.bns[i].load_state_dict(pretrain_model.encoder.bns[i].state_dict())

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-4)

print("\n🎯 Fine-tuning classification:")
classifier.train()
for epoch in tqdm(range(1, FINETUNE_EPOCHS+1), desc="Finetuning"):
    optimizer.zero_grad()
    logits = classifier(data.x, data.edge_index)  # already applies log_softmax
    mask   = (data.y != 2)  # ignore ‘unknown’ during train
    loss   = F.nll_loss(logits[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:03d}/{FINETUNE_EPOCHS} – Loss: {loss:.4f}")

# -----------------------------------------------------------------------------
# 6. Evaluate
# -----------------------------------------------------------------------------
classifier.eval()
with torch.no_grad():
    logits = classifier(data.x, data.edge_index)
    preds  = logits.argmax(dim=1)
    mask   = (data.y != 2)
    acc    = preds[mask].eq(data.y[mask]).float().mean().item()

print(f"\n✅ Final accuracy on known nodes: {acc*100:.2f}%")