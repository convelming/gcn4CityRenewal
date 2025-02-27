# **基于图结构的交通小区相似性分析**

项目总体思路为：
使用GCN处理网络--> 获取相似度高的node--> 根据OD矩阵做embedding 统一网络输入维度 训练重力模型网络-->
划定区域-->根据相似度高的node聚类生成candidate区域属性 -->使用训练好的重力模型多次生成产生与吸引 -->给出预测结果和上下限

## **步骤 0：基础数据的梳理**
所有数据基于OSM的路网结构，涉及到出生产生和吸引，需要OD矩阵或者OD对通过处理后（如小区层面的集计）作为样本数据。
在简化的情况下，只使用OSM数据提取每个节点的数据
## **步骤 1：定义不规则区域子图**
- 使用/src/visualization/drawPolygonOnOsm.html 在osm地图上绘制获取目标区域的坐标信息，双击结束绘制；

## **步骤 2：在整个交通网络中找到相似的区域**

## **步骤 3：计算相似性并排名**

## **步骤 4：训练重力图神经网络**

## **步骤 5：相似区域的属性替代目标区域并通过训练后的重力图神经网络预测出行分布**

## **步骤 6：MATSim 进行出行活动链建模**



梳理整个项目区域的路网（以广州地区为例），路网地图以open street map为主；空间属性为第三方开元或采购的数据，
## **步骤 1：定义不规则区域子图**
不规则区域子图的属性包括：
- **拓扑结构**：道路连接性、节点度分布、路段长度
- **空间属性**：POI（兴趣点）、AOI（兴趣区域）、道路类型、流量数据
- **功能属性**：土地利用、商业、居住、工业等

## **步骤 2：在整个交通网络中找到相似的区域**
### **方法 1：滑动窗口搜索相似子图**
```python
def get_k_hop_subgraph(G, center_node, k=2):
    subgraph_nodes = set([center_node])
    for _ in range(k):
        new_nodes = set()
        for node in subgraph_nodes:
            new_nodes.update(G.neighbors(node))
        subgraph_nodes.update(new_nodes)
    return G.subgraph(subgraph_nodes)
```
```python
candidate_subgraphs = {node: get_k_hop_subgraph(G, node, k=2) for node in G.nodes()}
```

## **步骤 3：计算相似性**
### **1. 计算拓扑相似度**
```python
import numpy as np

def degree_distribution_similarity(G1, G2):
    deg1 = np.array(sorted(dict(G1.degree()).values()))
    deg2 = np.array(sorted(dict(G2.degree()).values()))
    min_len = min(len(deg1), len(deg2))
    return np.linalg.norm(deg1[:min_len] - deg2[:min_len])
```

### **2. 计算 POI / AOI 相似性**
```python
from scipy.spatial.distance import jensenshannon
poi_similarity = 1 - jensenshannon(vector_1, vector_2)
```

### **3. 计算道路流量相似性**
```python
from scipy.stats import wasserstein_distance
traffic_similarity = 1 - wasserstein_distance(traffic_flow_1, traffic_flow_2)
```

## **步骤 4：综合计算相似性并排序**
```python
def compute_overall_similarity(G1, G2, poi1, poi2, flow1, flow2, alpha=0.4, beta=0.3, gamma=0.3):
    S_topo = 1 / (1 + degree_distribution_similarity(G1, G2))
    S_poi = 1 - jensenshannon(poi1, poi2)
    S_flow = 1 - wasserstein_distance(flow1, flow2)
    return alpha * S_topo + beta * S_poi + gamma * S_flow
```
```python
similarity_scores = {
    node: compute_overall_similarity(target_subgraph, subG, poi_data[node], poi_data['target'], flow_data[node], flow_data['target'])
    for node, subG in candidate_subgraphs.items()
}

sorted_similar_areas = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
```

## **步骤 5：后续分析**
- 可视化相似区域的道路网络、POI/AOI 分布、流量模式
- 计算它们在**整体城市结构中的分布**
- 结合交通规划优化相似区域的交通管理

## **总结**
1. **提取不规则目标子图**（空间查询 / `networkx`）
2. **搜索相似子图**（滑动窗口 / 社区检测）
3. **计算相似性**（拓扑结构、POI/AOI 分布、交通流量）
4. **排序并筛选最相似区域** 🚀
