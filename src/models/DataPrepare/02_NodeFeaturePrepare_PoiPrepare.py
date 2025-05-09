import geopandas as gpd
import pandas as pd
from src.models.DataPrepare.DataPrepareUtils import count_points_in_polygons,count_number_in_polygons

def PoiMerge(gpd_base,base_id='osmid'):#gpd_base为基础区域（1km网格/交通小区/泰森多边形），base_id为基础区域的id
    df_merge = pd.DataFrame()
    df_merge[base_id] = gpd_base[base_id]#用基础网格
    for poi_type in list_count_point_type:#导入不同类型的poi数据
        try:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp')
        except:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp',encoding='iso8859-1')
        df_count = count_points_in_polygons(gpd_points, gpd_base, base_id, poi_type)#计算在每个基础区域中有多少个poi点
        df_merge = pd.merge(df_merge,df_count,how='left',on=base_id)
    for poi_type in list_count_number_type:
        try:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp')
        except:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp',encoding='iso8859-1')
        df_count = count_number_in_polygons(gpd_points, gpd_base, base_id,dict_count_number_col[poi_type], poi_type)#计算在每个基础区域中有多少个poi中某属性总和
        df_merge = pd.merge(df_merge,df_count,how='left',on=base_id)
    return(df_merge)

if __name__ == '__main__':
    middle_data_floder = './src/models/middle_data/'
    base_data_floder = './src/models/base_data/'
    gpd_vor = gpd.read_file(middle_data_floder+'voronoi_gz.shp')#通过路网节点生成的泰森多边形
    list_count_point_type = ['poiBusStops','poiSubwayStations','buildingCentroids','poiCulture',
                         'poiGovFacility','poiHealthFacility','poiSport','poiEdu']#计算数量的POI
    list_count_number_type = ['poiParking']#计算属性总和的POI
    dict_count_number_col = {'poiParking':'总泊位'}#计算POI的哪一个属性
    df_result = PoiMerge(gpd_vor)
    df_result.to_csv(middle_data_floder+'base_poi.csv',index=False)


