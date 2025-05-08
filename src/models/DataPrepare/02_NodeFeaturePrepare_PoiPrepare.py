import geopandas as gpd
import pandas as pd
from src.models.DataPrepare.DataPrepareUtils import count_points_in_polygons,count_number_in_polygons

def PoiMerge(gpd_base,base_id='osmid'):
    df_merge = pd.DataFrame()
    df_merge[base_id] = gpd_base[base_id]
    for poi_type in list_count_point_type:
        try:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp')
        except:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp',encoding='iso8859-1')
        df_count = count_points_in_polygons(gpd_points, gpd_base, base_id, poi_type)
        df_merge = pd.merge(df_merge,df_count,how='left',on=base_id)
    for poi_type in list_count_number_type:
        try:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp')
        except:
            gpd_points = gpd.read_file(f'{base_data_floder}{poi_type}.shp',encoding='iso8859-1')
        df_count = count_number_in_polygons(gpd_points, gpd_base, base_id,dict_count_number_col[poi_type], poi_type)
        df_merge = pd.merge(df_merge,df_count,how='left',on=base_id)
    return(df_merge)

if __name__ == '__main__':
    middle_data_floder = './src/models/middle_data/'
    base_data_floder = './src/models/base_data/'
    gpd_vor = gpd.read_file(middle_data_floder+'voronoi_gz.shp')
    list_count_point_type = ['poiBusStops','poiSubwayStations','buildingCentroids','poiCulture',
                         'poiGovFacility','poiHealthFacility','poiSport','poiEdu']
    list_count_number_type = ['poiParking']
    dict_count_number_col = {'poiParking':'总泊位'}
    df_result = PoiMerge(gpd_vor)
    df_result.to_csv(middle_data_floder+'base_poi.csv',index=False)


