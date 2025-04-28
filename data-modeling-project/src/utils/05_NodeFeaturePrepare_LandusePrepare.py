import multiprocessing
import warnings
import os
import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")
area_id = 'osmid'
poi_path = '../data/base_gis/'
folder_path = '../data/base_data/landuse/'
# 检查文件夹是否存在
if not os.path.exists(folder_path):
    # 如果文件夹不存在，则创建文件夹
    os.makedirs(folder_path)

def read_file(shared_dict):
    gpd_fishnet = gpd.read_file('./base_data/voronoi_gz.shp')
    gpd_fishnet = gpd_fishnet.rename(columns={area_id: 'id'})
    landuse = gpd.read_file(poi_path+'landuse.shp')
    landuse = landuse.to_crs(gpd_fishnet.crs)
    landuse = landuse[['UUID','land_area','Level1_cn','geometry']]
    shared_dict['gpd_fishnet'] = gpd_fishnet
    shared_dict['landuse'] = landuse


def do_one(index, shared_dict):
    #for land in list(set(landuse['Level1_cn'])):
    landuse = shared_dict['landuse']
    gpd_fishnet = shared_dict['gpd_fishnet']
    all_num = len(gpd_fishnet)
    landuse_public = landuse[landuse['Level1_cn']==index]
    list_land =list()
    num = 0    
    for i in list(gpd_fishnet['id']):
        num +=1 
        a = gpd_fishnet[gpd_fishnet['id']==i]
        b = gpd.overlay(a,landuse_public,how='intersection')
        list_land.append(gpd.sjoin(b,a,'inner').area.values.sum())
        print(index,num/all_num)
    gpd_fishnet[index] = list_land
    gpd_cp = gpd_fishnet.copy()
    df_fishnet = pd.DataFrame(data = gpd_fishnet)
    df_fishnet = df_fishnet.rename(columns={'poiSubwayS': 'poiSubway'})
    df_fishnet = df_fishnet.rename(columns={'buildingCe': 'building'})
    df_fishnet = df_fishnet.rename(columns={'poiGovFaci': 'poiGov'})
    df_fishnet = df_fishnet.rename(columns={'poiHealthF': 'poiHealth'})
    df_fishnet = df_fishnet.rename(columns={'a_总人': 'total_pop'})
    df_fishnet = df_fishnet.rename(columns={'商业用地': 'business'})
    df_fishnet = df_fishnet.rename(columns={'交通用地': 'traffic'})
    df_fishnet = df_fishnet.rename(columns={'公共管理和服务用地': 'public'})
    df_fishnet = df_fishnet.rename(columns={'工业用地': 'industry'})
    df_fishnet = df_fishnet.rename(columns={'居住用地': 'resident'})
    gpd_fishnet = gpd.GeoDataFrame(df_fishnet, geometry=df_fishnet.geometry)
    gpd_fishnet.crs= gpd_cp.crs 
    gpd_fishnet.to_file('../data/base_data/landuse/base_landuse'+str(index)+'.shp', driver='ESRI Shapefile', encoding='utf-8')

def run_multi(shared_dict):
    # 读取基础数据
    read_file(shared_dict)
    # # 创建写入对象
    landuse = shared_dict['landuse']
    # 创建进程池参数
    inputs = [(i, shared_dict)
              for i in list(set(landuse['Level1_cn']))]
    print('input_finish')
    with multiprocessing.Pool(5) as pool:
        pool.starmap(do_one, inputs)    

if __name__ == '__main__':
    filepath = r"../data/base_data/landuse/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()
        run_multi(shared_dict)

    result = gpd.read_file('../data/base_data/voronoi_gz.shp')[[area_id]]
    result = result.rename(columns={area_id: 'id'})
    landuse = gpd.read_file(poi_path+'landuse.shp')
    for land_name in list(set(landuse['Level1_cn'])):
        df = gpd.read_file('../data/base_data/landuse/base_landuse'+str(land_name)+'.shp')
        del df['geometry']
        result = pd.merge(result,df)
    result = result.rename(columns={'id': area_id})
    result.to_csv('../data/base_data/base_landuse.csv', index=False)

