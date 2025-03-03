import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import subgraph
import torch.nn.functional as F
from collections import deque
import numpy as np


# 自定义图卷积层（考虑边特征）
class EdgeAwareGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_feature_dim):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels + edge_feature_dim, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.lin(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out):
        return aggr_out + self.bias


# 图卷积网络模型
class EdgeGCN(torch.nn.Module):
    def __init__(self, node_in, edge_in, hidden, out):
        super().__init__()
        self.conv1 = EdgeAwareGCNConv(node_in, hidden, edge_in)
        self.conv2 = EdgeAwareGCNConv(hidden, out, edge_in)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(x, edge_index, edge_attr)


# 生成k跳子图节点
def get_k_hop_nodes(center, k, edge_index, num_nodes):
    edge_index_np = edge_index.cpu().numpy()
    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in edge_index_np.T:
        adj_list[src].append(dst)

    visited = set()
    queue = deque([(center, 0)])
    nodes = set()
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        nodes.add(node)
        if depth < k:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
    return list(nodes)


# 示例图数据
num_nodes = 100
n_features = 32
e_features = 5
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.long)
x = torch.randn(num_nodes, n_features)
edge_attr = torch.randn(edge_index.size(1), e_features)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# 初始化模型
model = EdgeGCN(node_in=n_features, edge_in=e_features, hidden=64, out=32)

# 生成节点嵌入
with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index, data.edge_attr)

# 随机选择原区域
original_center = 0
k_hop = 2
original_nodes = get_k_hop_nodes(original_center, k_hop, data.edge_index, num_nodes)


# 计算原区域嵌入
def get_region_embedding(nodes, node_embeds, edge_index, edge_attr):
    sub_edge_index, sub_edge_attr = subgraph(nodes, edge_index, edge_attr, relabel_nodes=False)
    node_embed = node_embeds[nodes].mean(dim=0)
    edge_embed = sub_edge_attr.mean(dim=0) if sub_edge_attr.size(0) > 0 else torch.zeros(e_features)
    return torch.cat([node_embed, edge_embed])


original_embed = get_region_embedding(original_nodes, node_embeddings, data.edge_index, data.edge_attr)

# 寻找相似区域
candidates = []
np.random.seed(0)
for center in np.random.choice(num_nodes, 50, replace=False):
    if center == original_center:
        continue
    candidate_nodes = get_k_hop_nodes(center, k_hop, data.edge_index, num_nodes)
    if not candidate_nodes or set(candidate_nodes) & set(original_nodes):
        continue
    candidate_embed = get_region_embedding(candidate_nodes, node_embeddings, data.edge_index, data.edge_attr)
    similarity = F.cosine_similarity(original_embed.unsqueeze(0), candidate_embed.unsqueeze(0)).item()
    candidates.append((similarity, center, candidate_nodes))

# 输出最佳匹配
candidates.sort(reverse=True, key=lambda x: x[0])
if candidates:
    best_match = candidates[0]
    print(f"Best match - Center: {best_match[1]}, Similarity: {best_match[0]:.4f}")
    print(f"Original nodes: {original_nodes}")
    print(f"Matched nodes: {best_match[2]}")
else:
    print("No non-overlapping regions found.")