import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, method="laplacian", k=3, dropout=0.5):
        """
        通用 GCN 层
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param method: 选择图卷积方法（"laplacian", "spectrum", "chebyshev"）
        :param k: 切比雪夫多项式的阶数（仅在 method="chebyshev" 时生效）
        :param dropout: Dropout 概率
        """
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.k = k
        self.dropout = dropout

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化

    def forward(self, X, A):
        """
        前向传播
        :param X: 节点特征矩阵 (n, m)
        :param A: 归一化邻接矩阵 (n, n)
        :return: 更新后的特征
        """
        if self.method == "laplacian":
            out = self.laplacian_conv(X, A)
        elif self.method == "spectrum":
            out = self.spectral_conv(X, A)
        elif self.method == "chebyshev":
            out = self.chebyshev_conv(X, A, self.k)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # 应用 Dropout
        return F.dropout(out, self.dropout, training=self.training)

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
    def __init__(self, n, m, hidden_dim, output_dim, method="laplacian", k=3, dropout=0.5):
        """
        GCN 网络
        :param n: 节点数量
        :param m: 输入特征维度
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出特征维度
        :param method: 选择图卷积方式（"laplacian", "spectrum", "chebyshev"）
        :param k: 切比雪夫阶数（仅适用于 method="chebyshev"）
        :param dropout: Dropout 概率
        """
        super(GCN, self).__init__()
        self.conv1 = GraphConv(m, hidden_dim, method, k, dropout)
        self.conv2 = GraphConv(hidden_dim, output_dim, method, k, dropout)
        self.fc = nn.Linear(output_dim, 2)  # 输出 2 个连续值

    def forward(self, X, A):
        X = F.relu(self.conv1(X, A))
        X = self.conv2(X, A)
        X = self.fc(X)  # 前馈层
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

# ========== 定义模型 ==========
model = GCN(n, m, hidden_dim=8, output_dim=4, method="chebyshev", k=3, dropout=0.5)

# ========== 训练和验证 ==========
def train(model, X, A, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(X, A)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, X, A, labels, criterion):
    model.eval()
    with torch.no_grad():
        output = model(X, A)
        loss = criterion(output, labels)
    return loss.item()

# 假设真实标签是连续值（比如，回归任务）
labels = torch.randn(n, 2)  # 假设标签是二维连续值

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # 使用均方误差作为回归损失

# 训练和验证循环
for epoch in range(100):
    train_loss = train(model, X, A, labels, optimizer, criterion)
    val_loss = validate(model, X, A, labels, criterion)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, X, A, labels):
    """
    评估模型的预测结果
    :param model: 训练好的模型
    :param X: 节点特征矩阵
    :param A: 邻接矩阵
    :param labels: 真实标签
    :return: 各种评估指标
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 预测输出
        predictions = model(X, A)

        # 计算预测与实际标签的误差
        predictions = predictions.numpy()
        labels = labels.numpy()

        # 计算 MSE、MAE、RMSE 和 R²
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, predictions)

        return mse, mae, rmse, r2


# 示例用法：
# 假设我们有一个训练好的模型，X 和 A 是输入数据，labels 是真实标签
# 使用前面构造的数据进行评估：
mse, mae, rmse, r2 = evaluate_model(model, X, A, labels)

# 打印评估结果
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")