import ast

import geopandas as gpd
import numpy as np
import os

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import torch.nn as nn
from multiprocessing import Pool, cpu_count

models_floder = './src/models/'
class DynamicGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_num=3):
        super().__init__()
        self.conv_num = conv_num
        self.convs = nn.ModuleList()  # 动态存储所有GCN层

        # 添加输入层到第一隐藏层
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # 添加中间隐藏层（conv_num-2层）
        for _ in range(conv_num - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 添加最后一层到输出层
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 前conv_num-1层使用ReLU激活
        for i in range(self.conv_num - 1):
            x = F.relu(self.convs[i](x, edge_index))

        # 最后一层无激活函数
        x = self.convs[-1](x, edge_index)
        return x
def str_to_list(df,col_name):
    if type(df[col_name].values[0]) != list:
        df[col_name] =  df[col_name].apply(ast.literal_eval)
    return(df)


def load_data():
    """集中加载数据，避免重复I/O"""
    nodes = gpd.read_file(models_floder+'middle_data/guangzhou_drive_feature_node&edge.gpkg', layer='nodes')
    edges = gpd.read_file(models_floder+'middle_data/guangzhou_drive_feature_node&edge.gpkg', layer='edges')
    nodes = nodes.to_crs('EPSG:4526')
    nodes['x'] = nodes.geometry.x
    nodes['y'] = nodes.geometry.y
    return nodes, edges


def process_conv_num(conv_num, nodes, edges):
    """处理单个conv_num的任务"""
    print(f'Processing conv_num: {conv_num}')
    try:
        # 使用传入的数据副本进行处理
        df_nodes = nodes.copy()
        df_edges = edges.copy()
        t_data = gcn_features(df_nodes, df_edges, conv_num)
        df_nodes['features'] = [row.tolist() for row in t_data]
        df_nodes = df_nodes[['osmid', 'x', 'y', 'features']]
        output_path = f'{models_floder}middle_data/gcn_features/features_conv_{conv_num}.csv'
        df_nodes.to_csv(output_path, index=False)
        print(f'Finished conv_num: {conv_num}')
    except Exception as e:
        print(f'Error processing conv_num {conv_num}: {str(e)}')

def gcn_features(df_nodes,df_near,model_conv_num,nodeID_col='osmid',nodeFeatures_col='features',edgeSource_col='u',edgeTarget_col='v'):
    df_nodes = str_to_list(df_nodes,nodeFeatures_col)
    df_nodes = df_nodes[[nodeID_col,nodeFeatures_col]]
    node_id_to_idx = {id: idx for idx, id in enumerate(df_nodes[nodeID_col])}
    df_near['source_idx'] = df_near[edgeSource_col].map(node_id_to_idx)
    df_near['target_idx'] = df_near[edgeTarget_col].map(node_id_to_idx)
    x = torch.tensor(np.vstack(df_nodes[nodeFeatures_col]), dtype=torch.float)
    edge_index = torch.tensor(df_near[['source_idx', 'target_idx']].values.T, dtype=torch.long)
    # 添加反向边
    reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)  # 交换源和目标
    edge_index = torch.cat([edge_index, reverse_edges], dim=1)  # 拼接原边和反向边
    edge_index = add_self_loops(edge_index)[0]
    # 构建Data对象
    data = Data(x=x, edge_index=edge_index)
    try:
        model_layer = DynamicGCN(input_dim=20, hidden_dim=16, output_dim=20, conv_num=model_conv_num)
        model_layer.load_state_dict(torch.load(f'{models_floder}weights/gcn_weights_{model_conv_num}.pth'))
    except:
        model_layer = DynamicGCN(input_dim=20, hidden_dim=16, output_dim=20, conv_num=model_conv_num)
        torch.save(model_layer.state_dict(),f'{models_floder}weights/gcn_weights_{model_conv_num}.pth')
    model_layer.eval()
    with torch.no_grad():
        return model_layer(data).numpy()


def run_multi():
    # 预先加载数据
    nodes, edges = load_data()

    # 创建输出目录
    os.makedirs(f'{models_floder}middle_data/gcn_features', exist_ok=True)
    os.makedirs(f'{models_floder}weights', exist_ok=True)

    # 根据CPU核心数设置进程数
    num_processes = min(20, cpu_count())
    print(f'Using {num_processes} processes')

    # 使用进程池
    with Pool(num_processes) as pool:
        # 为每个conv_num创建任务
        tasks = [(i, nodes, edges) for i in range(1,21)]

        # 使用starmap传递多个参数
        pool.starmap(process_conv_num, tasks)


if __name__ == '__main__':
    run_multi()