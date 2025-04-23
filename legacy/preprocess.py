import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Define paths for the raw data files.
features_path = 'data/elliptic/elliptic_txs_features.csv'
edgelist_path = 'data/elliptic/elliptic_txs_edgelist.csv'
classes_path  = 'data/elliptic/elliptic_txs_classes.csv'
output_path   = 'data/elliptic/elliptic_data.pt'

# -------------------------------
# 1. Load the Node Features
# -------------------------------
features_df = pd.read_csv(features_path)
# Rename columns: first column is txId, remaining are features.
features_df.columns = ['txId'] + [f"V{i+1}" for i in range(features_df.shape[1] - 1)]

# Create a mapping from txId to node index (row number).
txid_to_idx = {txid: idx for idx, txid in enumerate(features_df['txId'])}

# Convert features (ignoring txId) to a torch tensor.
x = torch.tensor(features_df.drop('txId', axis=1).values, dtype=torch.float)
num_nodes = x.size(0)

# -------------------------------
# 2. Construct the Labels
# -------------------------------
# Load the classes file. (Assuming it has columns: 'txId' and 'class')
classes_df = pd.read_csv(classes_path)

# Create a dictionary mapping txId to label.
# We'll remap: if the CSV label is 1, assign 0 (e.g. illicit);
# if the CSV label is 2, assign 1 (e.g. licit).
# Nodes missing in the classes file are set as unknown (label 2).
class_map = {}
for _, row in classes_df.iterrows():
    txid = row['txId']
    cls = row['class']
    if cls == 1:
        label = 0
    elif cls == 2:
        label = 1
    try:
        label = int(cls)
    except ValueError:
        label = 2 
    class_map[txid] = label

# Build the label tensor for each node in the same order as features_df.
y_list = []
for txid in features_df['txId']:
    if txid in class_map:
        y_list.append(class_map[txid])
    else:
        # Unknown nodes are set to label 2.
        y_list.append(2)
y = torch.tensor(y_list, dtype=torch.long)

# -------------------------------
# 3. Build Edge Index from the Edgelist CSV
# -------------------------------
edges_df = pd.read_csv(edgelist_path)
# For this example, we assume the first two columns of the edgelist CSV are the source and target txIds.
if len(edges_df.columns) < 2:
    raise ValueError("Edgelist file must have at least two columns for the edge endpoints.")

src_col, dst_col = edges_df.columns[0], edges_df.columns[1]
# Map txIds to indices.
edge_src = edges_df[src_col].map(txid_to_idx)
edge_dst = edges_df[dst_col].map(txid_to_idx)
# Remove any rows where mapping failed.
mask = edge_src.notnull() & edge_dst.notnull()
edge_src = edge_src[mask].astype(int).values
edge_dst = edge_dst[mask].astype(int).values

# Build edge_index tensor.
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
# Make the graph undirected.
edge_index = to_undirected(edge_index, num_nodes=num_nodes)

# -------------------------------
# 4. Create the PyTorch Geometric Data Object
# -------------------------------
data = Data(x=x, y=y, edge_index=edge_index)
data.num_nodes = num_nodes  # This is optional since x already determines num_nodes.

# -------------------------------
# 5. Create Train/Test Splits on Known Nodes
# -------------------------------
# Known nodes: those with y != 2.
known_mask = (data.y != 2)
known_indices = torch.where(known_mask)[0]
# Shuffle the known indices.
perm = known_indices[torch.randperm(len(known_indices))]
split_idx = int(0.8 * len(perm))
train_indices = perm[:split_idx]
test_indices = perm[split_idx:]

# Initialize boolean masks for all nodes.
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_indices] = True
data.test_mask[test_indices] = True

print("Data loaded successfully. Number of nodes:", data.num_nodes)
print(f"Train nodes: {data.train_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")

# -------------------------------
# 6. Save the Data Object
# -------------------------------
torch.save(data, output_path)
print(f"Elliptic data saved to {output_path}")
