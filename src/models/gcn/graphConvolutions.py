import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, method="laplacian", k=3):
        """
        通用 GCN 层
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param method: 选择图卷积方法（"laplacian", "spectrum", "chebyshev"）
        :param k: 切比雪夫多项式的阶数（仅在 method="chebyshev" 时生效）
        """
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.k = k

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化

    def forward(self, X, A):
        """
        前向传播
        :param X: 节点特征矩阵 (n, m)
        :param A: 归一化邻接矩阵 (n, n)（需要提前处理）
        :return: 更新后的特征
        """
        if self.method == "laplacian":
            return self.laplacian_conv(X, A)
        elif self.method == "spectrum":
            return self.spectral_conv(X, A)
        elif self.method == "chebyshev":
            return self.chebyshev_conv(X, A, self.k)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def laplacian_conv(self, X, A):
        """拉普拉斯图卷积"""
        return torch.spmm(A, X).mm(self.weight)

    def spectral_conv(self, X, A):
        """谱方法图卷积（特征分解）"""
        eigenvalues, eigenvectors = torch.linalg.eigh(A)  # A 的特征分解
        spectral_A = eigenvectors @ torch.diag_embed(F.relu(eigenvalues)) @ eigenvectors.T
        return torch.spmm(spectral_A, X).mm(self.weight)

    def chebyshev_conv(self, X, A, k):
        """切比雪夫多项式图卷积"""
        n = A.shape[0]
        I = torch.eye(n, device=A.device)
        L_tilde = 2 * A - I  # 归一化的拉普拉斯
        T_k = [I, L_tilde]  # 存储切比雪夫多项式的 T0 和 T1
        for i in range(2, k):
            T_k.append(2 * torch.spmm(L_tilde, T_k[-1]) - T_k[-2])
        output = sum(torch.spmm(T_k[i], X).mm(self.weight) for i in range(k))
        return output

class GCN(nn.Module):
    def __init__(self, n, m, hidden_dim, output_dim, method="laplacian", k=3):
        """
        GCN 网络
        :param n: 节点数量
        :param m: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出特征维度
        :param method: 选择图卷积方式（"laplacian", "spectrum", "chebyshev"）
        :param k: 切比雪夫阶数（仅适用于 method="chebyshev"）
        """
        super(GCN, self).__init__()
        self.conv1 = GraphConv(m, hidden_dim, method, k)
        self.conv2 = GraphConv(hidden_dim, output_dim, method, k)

    def forward(self, X, A):
        X = F.relu(self.conv1(X, A))
        X = self.conv2(X, A)
        return X

# ========== 构造测试数据 ==========
def normalize_adj(A):
    """计算归一化邻接矩阵"""
    A = A + sp.eye(A.shape[0])  # 加上自环
    D = np.array(A.sum(1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    return torch.FloatTensor(D_inv_sqrt @ A @ D_inv_sqrt)

# 随机构造邻接矩阵和节点特征
n, m = 5, 3  # 5个节点，每个节点3个特征
A_np = np.array([[0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1],
                 [1, 1, 0, 1, 0]], dtype=np.float32)

X_np = np.random.rand(n, m)  # 随机特征矩阵

A = normalize_adj(A_np)
X = torch.FloatTensor(X_np)

# ========== 运行 GCN ==========
model = GCN(n, m, hidden_dim=8, output_dim=2, method="chebyshev", k=3)
output = model(X, A)
print(output)
