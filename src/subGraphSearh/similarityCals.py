'''
In this script, multiple stardards are used to calculate subgraph similarities.
Then definition of total evaluation between two sub-graphs is given, and it is basically sum of weighted similarities defined here.

1. 多维相似度指标
1-1结构相似性：
1-1-1 度分布KL散度：DKL(Pa∥Pb)=∑ d log (Pa(d)Pb(d))
1-1-2 路径特征：平均最短路径长度比
1-1-3 聚类系数差异：

1-2 属性相似性：
1-2-1 节点特征余弦相似度矩阵：
1-2-2 边属性匹配率：


1-3 动态行为相似性（暂时不考虑）：

1-3-1 信息传播模式对比（使用SIR模型参数）
1-3-2 时序交互频率分布

2. 相似度聚合函数
通过加权平均的方式计算
Similarity=∏ sigma(w_k*s_k)
'''
import math

import numpy as np

import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def cal_graph_degree_distribution(graph, direction="in"):
    """
    计算有向图的度分布（入度或出度）。
    :param graph: NetworkX DiGraph
    :param direction: "in" 表示入度, "out" 表示出度
    :return: 度概率分布 (dict)
    """
    if direction == "in":
        degrees = [graph.in_degree(n) for n in graph.nodes()]
    else:
        degrees = [graph.out_degree(n) for n in graph.nodes()]

    # 计算度的概率分布
    total_nodes = len(degrees)
    degree_count = {d: degrees.count(d) / total_nodes for d in set(degrees)}

    return degree_count


def kl_divergence(P, Q, epsilon=1e-10):
    """
    计算 KL 散度: D_KL(P || Q)
    :param P: 目标分布 (dict)
    :param Q: 参考分布 (dict)
    :param epsilon: 防止 log(0) 的小数
    :return: KL 散度值
    """
    # 取得所有可能的度数 d
    all_keys = set(P.keys()).union(set(Q.keys()))

    kl_sum = 0.0
    for d in all_keys:
        p_d = P.get(d, epsilon)  # 避免 log(0)
        q_d = Q.get(d, epsilon)  # 避免 log(0)
        kl_sum += p_d * np.log(p_d / q_d)

    return kl_sum


def cal_KL_divergence(graph_1, graph_2, in_weight=0.5):
    """
    Kullback-Leibler Divergence KL 散度
        KL散度用于衡量两个概率分布P 和 Q之间的差异。在度分布的场景中：P 表示子图 graph_1的度分布概率；Q表示graph_2的度分布概率子图，则
        D_kL(P||Q) = sum_d P(d)log(P(d)/Q(d))
        其中d为节点度数，
        P（d），Q（d）为其图中度数为d的节点数/图中的总节点数
        由于路网时有向图 所有计算出来的KL散度需要考虑双向 dk_bi = dir_weight*dk + (1-dir_weight)*dk
        KL 值越小 表示两个图的的节点分布越接近，即评估时越小越好
        :param graph_1:  NetworkX DiGraph
        :param graph_2:  NetworkX DiGraph
        :param dir_weight: weight for the in
        :return:
    """

    # 计算入度和出度分布
    in_degree_dist_1 = cal_graph_degree_distribution(graph_1, "in")
    out_degree_dist_1 = cal_graph_degree_distribution(graph_1, "out")
    in_degree_dist_2 = cal_graph_degree_distribution(graph_2, "in")
    out_degree_dist_2 = cal_graph_degree_distribution(graph_2, "out")

    # 计算入度和出度的 KL 散度
    kl_in = kl_divergence(in_degree_dist_1, in_degree_dist_2)
    kl_out = kl_divergence(out_degree_dist_1, out_degree_dist_2)

    # 计算最终的双向 KL 散度
    kl_bi = in_weight * kl_in + (1 - in_weight) * kl_out
    return kl_bi

def compute_aspl(graph):
    """
    计算有向图的平均最短路径长度 (ASPL)
    :param graph: networkx.DiGraph
    :return: 平均最短路径长度 ASPL
    """
    total_length = 0
    valid_pairs = 0  # 记录可达的节点对

    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, d_ij in lengths.items():
            if source != target:
                total_length += d_ij
                valid_pairs += 1

    return total_length / valid_pairs if valid_pairs > 0 else float('inf')

def cal_shortest_path_length_ratio(graph_1, graph_2):
    """
    1-1-2 路径特征：平均最短路径长度比
    平均最短路径长度（ASPL） 是子图中所有节点对之间最短路径长度的平均值。对于两个子图 graph_1与grahp_2 其平均最短路径长度比定义为：
    路径长度比 = ASPL_graph1/APL_graph2
    ASPL = 1/(N*(N-1)) sum_(i!=j) d_ij
    d_ij 表示节点 i 到 j 的最短路径长度（边数），N为子图节点数
    排除不连通对：若 i 到 j 不可达，可忽略该对或设 d_ij=infinity，但需在分母中扣除无效对,所以：
    ASPL  = sum_(i!=j) d_ij / 有效节点对数

    若结果接近 1，说明两者连通性相似；显著偏离 1 则表明结构差异。

    针对这个问题需要将这个比率归一化极倔0.9和1.1谁的连通性更相似，采用1/(1+abs(log(ASPL_ratio)))这个归一化方法的优点：
    对称性：ASPL 比值大于 1 和小于 1 都会被同等对待。
    范围约束：归一化值始终在 (0, 1] 之间。
    可解释性：当比值为 1 时，归一化值最大为 1；比值越偏离 1，归一化值越小。

    :param graph_1: di-graph from networkX
    :param graph_2: di-graph from networkX
    :return:  ASPL_g1 / ASPL_g2
    """
    aspl_g1 = compute_aspl(graph_1)
    aspl_g2 = compute_aspl(graph_2)

    if aspl_g2 == 0 or aspl_g2 == float('inf'):  # 避免除零错误
        return float('inf')

    return 1 / (1+math.log(aspl_g1 / aspl_g2))

def cal_cluster_coe_diff(graph_1, graph_2):
    """
    计算 聚类系数差异（Clustering Coefficient Difference）

    平均局部聚类系数：C_avg = 1/N * Sum C_i
    N 为节点数量
    C_i = 节点i的邻居间实际边数/邻居间可能的边数
    对每个节点 i，计算其邻居间实际边数E_i和可能的最大边数k_i(k_i-1) k_i>=2
    C_i = 2*E_i/(k_i*(k_i-1))
    所有节点的C_i取平均获取C_graph1，C_graph2；
    两者差的绝对值即为聚类系数差异 = |C_graph1，C_graph2|

    :param graph_1: di-graph from networkX
    :param graph_2: di-graph from networkX
    :return: |C_graph1-C_graph2|
    """

    C_graph1 = nx.average_clustering(graph_1.to_undirected())
    C_graph2 = nx.average_clustering(graph_2.to_undirected())

    return abs(C_graph1 - C_graph2)


def cal_graph_cosine_similarity(graph_1, graph_2, attr_key='features', sim_type="max_pooling"):
    """
    图节点特征余弦相似度矩阵：
    对于两个子图graph_1(m个节点), graph_2(n个节点)其节点特征矩阵分别为X_a(mxd), X_b(nxd)(d为特征维度)
    余弦相似度矩阵计算为： S_node = X_a* X_b^T/||X_a||*||X_b||
    ∥X∥ 为按行L2范数归一化后的矩阵（即每行特征向量归一化为单位长度）
    步骤1：特征归一化
    X_A_normalized = X_A / np.linalg.norm(X_A, axis=1, keepdims=True)
    X_B_normalized = X_B / np.linalg.norm(X_B, axis=1, keepdims=True)
    步骤2：矩阵乘法
    cos_sim_matrix = np.dot(X_A_normalized, X_B_normalized.T)
    步骤3：汇总为子图相似度
    全连接匹配平均（适用于节点数相同且对齐的场景）full_conn_avg：1/m * sum sii
    最大池化匹配（适用于节点数不同或无序场景）max_pooling S_node=1/m * sum_m max (s_ij) where i 1 to m, j 1 to n
    最优传输匹配（考虑全局最优对齐）匈牙利算法 global_opt：
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix)  # 最大化相似度
        S_node = cos_sim_matrix[row_ind, col_ind].mean()
    :param graph_1:
    :param graph_2:
    :return:
    结果接近1表示两图节点特征匹配良好
    """
    # 获取图节点特征
    X_a = np.array([graph_1.nodes[i][attr_key].flatten() for i in graph_1.nodes])
    X_b = np.array([graph_2.nodes[i][attr_key].flatten() for i in graph_2.nodes])

    # 归一化节点特征（L2 归一化）
    X_a_normalized = X_a / np.linalg.norm(X_a, axis=1, keepdims=True)
    X_b_normalized = X_b / np.linalg.norm(X_b, axis=1, keepdims=True)

    # 计算余弦相似度矩阵 (m×n)
    cos_sim_matrix = np.dot(X_a_normalized, X_b_normalized.T)

    # 根据不同匹配策略计算相似度
    if sim_type == "full_conn_avg":
        if X_a.shape[0] != X_b.shape[0]:
            raise ValueError("full_conn_avg 只适用于两个子图的节点数相等的情况")
        S_node = np.mean(np.diag(cos_sim_matrix))  # 对角线均值

    elif sim_type == "max_pooling":
        S_node = np.mean(np.max(cos_sim_matrix, axis=1))  # 每个节点取最大相似度

    elif sim_type == "global_opt":
        row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix)  # 最大化匹配
        S_node = cos_sim_matrix[row_ind, col_ind].mean()  # 计算匹配的均值

    else:
        raise ValueError("sim_type 只能是 'full_conn_avg', 'max_pooling' 或 'global_opt'")

    return S_node


def cal_edge_similarity(graph_1, graph_2, attr_key="volumes", match_strategy="max_pooling"):
    """
    计算两个图边的属性的相似性,此方法不考虑节点特征和图结构的
    步骤1：边属性对齐, 边的数量分别为m，n
    图为有向图：需匹配边的方向（即(u, v)与(p, q)严格对应）
    步骤2：属性匹配计算 边edge_i的属性为 nx1的属性，通常为24小时流量，即n=24
    为连续值属性，但需要注意的是可能有空值；
    步骤3：结果归一化，因为为连续属性，需根据相似度函数调整范围

    :param graph_1: nx.DiGraph - 第一个有向图
    :param graph_2: nx.DiGraph - 第二个有向图
    :param attr_key: str - 需要计算相似度的边属性（默认 'volumes'）
    :return: float - 余弦相似度（范围 [0,1]，越接近1表示越相似）
    """
    # 获取两个图的边集合
    edges_1 = list(graph_1.edges(data=True))
    edges_2 = list(graph_2.edges(data=True))

    # 提取所有边的属性值（如果没有流量属性，则默认为 0 向量）
    edge_features_1 = np.array([data.get(attr_key, np.zeros(24)) for _, _, data in edges_1])
    edge_features_2 = np.array([data.get(attr_key, np.zeros(24)) for _, _, data in edges_2])

    # 处理 NaN 值（用均值填充）
    mean_1 = np.nanmean(edge_features_1, axis=0) if np.isnan(edge_features_1).any() else 0
    mean_2 = np.nanmean(edge_features_2, axis=0) if np.isnan(edge_features_2).any() else 0
    edge_features_1 = np.nan_to_num(edge_features_1, nan=mean_1)
    edge_features_2 = np.nan_to_num(edge_features_2, nan=mean_2)

    # 计算余弦相似度矩阵 (m × n)
    cos_sim_matrix = cosine_similarity(edge_features_1, edge_features_2)

    # 选择匹配策略
    if match_strategy == "full_conn_avg":
        # 适用于边数量相同且对齐的情况
        if edge_features_1.shape[0] != edge_features_2.shape[0]:
            raise ValueError("full_conn_avg 适用于两个图的边数相同的情况")
        similarity = np.mean(np.diag(cos_sim_matrix))

    elif match_strategy == "max_pooling":
        # 适用于边数量不同的情况，每条边找最相似的匹配
        similarity = np.mean(np.max(cos_sim_matrix, axis=1))

    elif match_strategy == "global_opt":
        # 适用于全局最优匹配，使用匈牙利算法
        row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix)  # 负号表示最大化相似度
        similarity = cos_sim_matrix[row_ind, col_ind].mean()

    else:
        raise ValueError("match_strategy 只能是 'full_conn_avg', 'max_pooling' 或 'global_opt'")

    return similarity


def cal_total_weighted_similarity(graph_1, graph_2, weights={'kl': 0.2, 'sp': 0.2, 'cluster': 0.2, 'node': 0.2, 'edge': 0.2},
                                  node_attr_key="features", edge_attr_key='volumes', show=False):

    kl_div = cal_KL_divergence(graph_1, graph_2)
    sp_ratio = cal_shortest_path_length_ratio(graph_1, graph_2)
    cluster_diff = cal_cluster_coe_diff(graph_1, graph_2)

    node_similarity = cal_graph_cosine_similarity(graph_1, graph_2, attr_key=node_attr_key)
    edge_similarity = cal_edge_similarity(graph_1, graph_2, attr_key=edge_attr_key)


    similarity = (weights['kl'] * kl_div +
                  weights['sp'] * sp_ratio +
                  weights['cluster'] * cluster_diff +
                  weights['node'] * node_similarity +
                  weights['edge'] * edge_similarity)

    if show:
        print(
            f"KL divergence: {kl_div:.4f},\n"
            f" short path length Ratio: {sp_ratio:.4f},\n"
            f" Cluster Diff: {cluster_diff:.4f},\n"
            f" Node Similarity: {node_similarity:.4f},\n"
            f" Edge Similarity: {edge_similarity:.4f}")
        print(f"Final Similarity Score: {similarity:.4f}")

    return similarity

# 1-3 动态行为相似性（可选）：
#
# 1-3-1 信息传播模式对比（使用SIR模型参数）
# 1-3-2 时序交互频率分布








