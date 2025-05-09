import random
import geopandas as gpd

import osmnx as ox
import pandas as pd

import pickle

from multiprocessing import Pool, Manager, Lock
from src.models.DataPrepare.InputPrepareUtils import get_subgraph_list,creat_input


middle_data_floder = './src/models/middle_data/'
TrainData_floder = './src/models/TrainData/'


#多进程生成训练样本
file_lock = Lock()
def read_file():
    # 每个进程独立加载数据，避免共享大数据
    r_graph = ox.load_graphml(middle_data_floder+'guangzhou_drive_feature_node&edge.graphml') #加载网络
    r_gdf_node = gpd.read_file(middle_data_floder+'guangzhou_drive_feature_node&edge.gpkg', layer='nodes') #加载节点属性
    r_gdf_node = r_gdf_node.to_crs('EPSG:4526')
    r_gdf_node['x'] = r_gdf_node.apply(lambda z: z.geometry.x, axis=1)
    r_gdf_node['y'] = r_gdf_node.apply(lambda z: z.geometry.y, axis=1)
    r_df_od = pd.read_csv(middle_data_floder+'base_od.csv') #加载OD量
    return r_graph, r_gdf_node, r_df_od

def do_one(index, result_queue):
    print(f'kernel_{index}_start')
    try:
        shared_graph, shared_gdf_node, shared_df_od = read_file()
        list_search_kernel = list()
        df_xy_kernel = pd.DataFrame()
        while len(list_search_kernel) < 200: #单次生成数量停止条件
            random_osmid = random.choice(shared_gdf_node['osmid']) #随机搜索中心点
            random_depth = random.randint(4, 20)             #在4到20间随机搜索中心子图深度
            random_id = f"{int(random_osmid)}_{int(random_depth)}"
            if random_id not in list_search_kernel: #判断该中心点及其深度是否搜索过
                subgraph_list = get_subgraph_list(shared_graph, random_depth, random_osmid)
                df_xy = creat_input(shared_gdf_node,shared_df_od,subgraph_list) #创建训练样本
                df_xy = df_xy[['input_x','flowcounts']]

                df_xy_kernel = pd.concat([df_xy_kernel, df_xy])#合并样本
                list_search_kernel.append(random_id)
                if len(list_search_kernel) % 1 == 0:
                    # 将结果放入队列而不是直接写文件
                    result_queue.put((list_search_kernel.copy(), df_xy_kernel.copy()))
                    df_xy_kernel = pd.DataFrame()  # 清空临时DataFrame
        # 处理剩余未提交的结果
        if not df_xy_kernel.empty:
            result_queue.put((list_search_kernel, df_xy_kernel))
    except Exception as e:
        print(f"Process {index} failed: {str(e)}")


def result_writer(result_queue):
    """专门负责写文件的进程"""
    list_search = []
    df_all_xy = pd.DataFrame()
    try:
        # 尝试加载已有数据
        with file_lock:
            try:
                with open(TrainData_floder+'list_search.pkl', "rb") as f:
                    list_search = pickle.load(f)
            except FileNotFoundError:
                pass
            try:
                df_all_xy = pd.read_csv(TrainData_floder+'graph_kmeans_input_data.csv')
            except FileNotFoundError:
                pass
    except Exception as e:
        print(f"Error loading existing data: {str(e)}")
    while True:
        try:
            new_list, new_df = result_queue.get()
            if new_list == "DONE":  # 结束信号
                break
            # 合并新数据
            list_search = list(set(list_search + new_list))
            new_df['input_x'] = new_df['input_x'].apply(str)
            df_all_xy = pd.concat([df_all_xy, new_df]).drop_duplicates()
            # 写入文件
            with file_lock:
                with open(TrainData_floder+'list_search.pkl', "wb") as f:
                    pickle.dump(list_search, f)
                df_all_xy.to_csv(TrainData_floder+'graph_kmeans_input_data.csv', index=False)
            print(f"Writer: Current total records: {len(df_all_xy)}")
        except Exception as e:
            print(f"Error in writer process: {str(e)}")


def run_multi():
    # 使用队列收集结果
    manager = Manager()
    result_queue = manager.Queue()
    # 启动写入进程
    writer_pool = Pool(1)
    writer_pool.apply_async(result_writer, (result_queue,))
    # 启动工作进程
    worker_pool = Pool(20)
    for i in range(20):
        worker_pool.apply_async(do_one, (i, result_queue))
    worker_pool.close()
    worker_pool.join()
    # 通知写入进程结束
    result_queue.put(("DONE", None))
    writer_pool.close()
    writer_pool.join()


if __name__ == '__main__':
    run_multi()    