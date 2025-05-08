import os
import pandas as pd
import multiprocessing
import geopandas as gpd


area_id = 'osmid'
poi_path = './src/models/base_data/'
middle_data_floder = './src/models/middle_data/'

# 检查文件夹是否存在
if not os.path.exists(middle_data_floder+'road/'):
    # 如果文件夹不存在，则创建文件夹
    os.makedirs(middle_data_floder+'road/')

def read_file(shared_dict):
    gpd_fishnet = gpd.read_file(middle_data_floder+'voronoi_gz.shp')
    gpd_fishnet = gpd_fishnet.rename(columns={area_id: 'id'})
    network = gpd.read_file(poi_path+'network.shp', encoding='iso8859-1')
    network = network[['osm_id','highway','geometry']].dropna()
    all_num = len(gpd_fishnet)

    shared_dict['gpd_fishnet'] = gpd_fishnet
    shared_dict['network'] = network
    shared_dict['all_num'] = all_num

def cal_road(gid,gpd_data,network_data):
    a = gpd_data[gpd_data['id']==gid]
    b = gpd.overlay(network_data,a,how='intersection')
    road_len =gpd.sjoin(b,a,'inner').length.values.sum()    
    return(road_len)

def do_one(index,road_type,step,shared_dict):

    gpd_fishnet = shared_dict['gpd_fishnet']
    end = len(gpd_fishnet) if index+step >len(gpd_fishnet) else index+step
    network = shared_dict['network']
    list_main = ['trunk','primary','motorway','secondary','motorway_link','trunk_link','primary_link','secondary_link']
    list_residential = ['service','residential','tertiary','tertiary_link','living_street']
    list_other = list(set(network ['highway'])-set(list_main)-set(list_residential))
    dict_road = {'main':list_main,'residential':list_residential,'other': list_other}
    network_main = network[network['highway'].isin(dict_road[road_type])]

    data = gpd_fishnet[index:end]
    data[road_type] = data.apply(lambda z:cal_road(z.id,data,network_main),axis=1)
    data = data[['id',road_type]]
    print(road_type,index,'cal')

    data.to_csv(middle_data_floder+'road/'+str(road_type)+'/'+str(index)+'.csv', index=False, encoding='utf-8')

def run_multi(shared_dict,road_type):
    for road_type in ['main','residential','other']:
        all_num = len(gpd.read_file(middle_data_floder+'voronoi_gz.shp'))
        step = 50
        folder = middle_data_floder+'road/' + str(road_type)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        list_fol = [int(int(s.replace('.csv', '')) / step) for s in os.listdir(folder)]
        list_all = list(range(int(all_num/step)+1))
        diff = list(set(list_all) - set(list_fol))
        read_file(shared_dict)
        inputs = [(i * step, road_type, step, shared_dict)
                  for i in diff]
        print('input_finish')
        with multiprocessing.Pool(10) as pool:
            pool.starmap(do_one, inputs)


if __name__ == '__main__':
    #lock = multiprocessing.Lock()
    for road_type in ['main','residential','other']:
        filepath = middle_data_floder+'/road/' +road_type+'/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()
        run_multi(shared_dict,road_type)

    gpd_result = gpd.read_file(middle_data_floder+'voronoi_gz.shp')[[area_id]]
    gpd_result = gpd_result.rename(columns={area_id: 'id'})
    for road_type in ['main','residential','other']:
        df_all = pd.DataFrame()
        folder = middle_data_floder+'road/'+str(road_type)
        for file_name in os.listdir(folder):
            df = pd.read_csv(folder+'/'+file_name)
            df_all = pd.concat([df_all,df])
        gpd_result = pd.merge(gpd_result,df_all)
    gpd_result = gpd_result.rename(columns={'id': area_id})
    gpd_result.to_csv(middle_data_floder+'base_road.csv',index=False)



