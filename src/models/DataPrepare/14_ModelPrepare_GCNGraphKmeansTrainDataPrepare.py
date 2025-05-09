import random
import geopandas as gpd
import osmnx as ox
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from multiprocessing import Pool, Manager, Lock
from src.models.DataPrepare.InputPrepareUtils import creat_input,get_subgraph_list


TrainData_floder = './src/models/TrainData/'
middle_data_floder = './src/models/middle_data/'


#多进程生成训练样本
file_lock = Lock()
def read_file(conv_num):
    # 每个进程独立加载数据，避免共享大数据
    if conv_num == 1:
        r_gdf_node = gpd.read_file(f'{middle_data_floder}guangzhou_drive_feature_node&edge.gpkg', layer='nodes') #加载节点属性
        r_gdf_node = r_gdf_node.to_crs('EPSG:4526')
        r_gdf_node['x'] = r_gdf_node.apply(lambda z: z.geometry.x, axis=1)
        r_gdf_node['y'] = r_gdf_node.apply(lambda z: z.geometry.y, axis=1)
    else:
        r_gdf_node = pd.read_csv(f'{middle_data_floder}gcn_features/features_conv_{conv_num}.csv')
    r_graph = ox.load_graphml(f'{middle_data_floder}guangzhou_drive_feature_node&edge.graphml') #加载网络
    r_df_od = pd.read_csv(f'{middle_data_floder}base_od.csv') #加载OD量
    return r_graph, r_gdf_node, r_df_od

def do_one(index):
    print(f'kernel_{index}_start')
    shared_graph, shared_gdf_node, shared_df_od = read_file(index)
    list_search_kernel = list()
    df_xy_kernel = pd.DataFrame()
    while len(df_xy_kernel) < 20_0000: #单次生成数量停止条件
        random_osmid = random.choice(shared_gdf_node['osmid']) #随机搜索中心点
        random_depth = random.randint(4, 20)             #在4到20间随机搜索中心子图深度
        random_id = f"{int(random_osmid)}_{int(random_depth)}"
        if random_id not in list_search_kernel: #判断该中心点及其深度是否搜索过
            subgraph_list = get_subgraph_list(shared_graph, random_depth, random_osmid)
            df_xy = creat_input(shared_gdf_node, shared_df_od, subgraph_list)  # 创建训练样本
            df_xy = df_xy[['input_x', 'flowcounts']]
            df_xy_kernel = pd.concat([df_xy_kernel, df_xy])#合并样本
            list_search_kernel.append(random_id)
            print(f'kernel_{index}_len_{len(df_xy_kernel)}')
    df_xy_kernel.to_csv(f'{TrainData_floder}graph_kmeans_input_data_{index}.csv', index=False)

def run_multi():
    # 使用队列收集结果
    manager = Manager()
    # result_queue = manager.Queue()
    # 启动工作进程
    list_conv_num = [20]
    worker_pool = Pool(len(list_conv_num))
    for i in list_conv_num:
        worker_pool.apply_async(do_one, (i, ))
    worker_pool.close()
    worker_pool.join()

if __name__ == '__main__':
    run_multi()    