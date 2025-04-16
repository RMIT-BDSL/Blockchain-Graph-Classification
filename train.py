#!/usr/bin/env python
# coding: utf-8

# ## Data Exploration

# In[1]:


import networkx as nx
import pandas as pd
import gc
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch_geometric.data import Data
from model.GCN import GCNClassifier


# In[2]:


edgelists = pd.read_csv('data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
features = pd.read_csv('data/elliptic_bitcoin_dataset/elliptic_txs_features.csv')
classes = pd.read_csv('data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')


# In[3]:


features.columns = ['txId'] + [f"V{i + 1}" for i in range(len(features.columns) - 1)]


# In[4]:


print("Number of nodes:", len(features))
print("Number of edges:", len(edgelists))


# In[5]:


classes['class_mapped'] = classes['class'].replace({'1': 'illicit', '2': 'licit'})

percentage_distribution = round(100 * classes['class_mapped'].value_counts(normalize=True), 2)
class_counts = classes['class_mapped'].value_counts()

emoji_mapping = {
    'licit': '✅', 
    'illicit': '❌', 
    'unknown': '🤷'
}
classes['emoji'] = classes['class_mapped'].map(emoji_mapping)

classes_df = pd.DataFrame({
    'Class Mapped': classes['class_mapped'].unique(),
    'Class Raw': classes['class'].unique(),    
    'Counts': class_counts.values,
    'Percentage': percentage_distribution.values,
    'Emoji': [emoji_mapping[class_label] for class_label in classes['class_mapped'].unique()]
})

assert len(classes_df) == 3, "There should be 3 unique classes"
assert sum(classes_df['Counts']) == len(classes), "Total counts should match the number of rows in classes"


# In[6]:


classes_df


# ## Loading preprocessed data

# In[7]:


with torch.serialization.safe_globals([Data]):
    data = torch.load('data/elliptic_bitcoin_dataset/elliptic_data.pt', weights_only=False)

print("Data loaded successfully. Number of nodes:", data.num_nodes)


# ## Model Initialization

# In[8]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNClassifier(data.num_node_features, hidden_dim=64, num_classes=3).to(device)
data = data.to(device)


# ## Model training

# In[9]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(1, 101), desc="Training"):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    mask = (data.y != 2)  # ignore 'unknown'
    loss = F.nll_loss(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


# In[10]:


model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    known_mask = (data.y != 2)
    correct = preds[known_mask].eq(data.y[known_mask]).sum().item()
    acc = correct / known_mask.sum().item()
    print(f"Accuracy on known-labeled nodes: {acc*100:.2f}%")

