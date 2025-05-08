import geopandas as gpd
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

area_id = 'osmid'
middle_data_floder = './src/models/middle_data/'

gpd_vor = gpd.read_file(middle_data_floder+'voronoi_gz.shp')
def count_points_in_polygons(gpd_points,gpd_polygons,polygon_col,new_col_name):
    gpd_points = gpd_points[['geometry']].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = pd.DataFrame()
    df_vc['fishid'] = df_join[polygon_col].value_counts().index
    df_vc[new_col_name] = df_join[polygon_col].value_counts().values
    return(df_vc)

def count_number_in_polygons(gpd_points,gpd_polygons,polygon_col,cal_num_col,new_col_name):
    gpd_points = gpd_points[['geometry',cal_num_col]].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = df_join.groupby([polygon_col])[cal_num_col].sum().reset_index()
    df_vc.columns = ['fishid',new_col_name]
    return(df_vc)

poi_path = './src/models/base_data/'
gpd_fishnet = gpd_vor.copy()
gpd_fishnet = gpd_fishnet.rename(columns={area_id: 'id'})
df_fishnet = pd.DataFrame()
df_fishnet['fishid'] = gpd_fishnet['id']
gpd_points = gpd.read_file(poi_path+'poiBusStops.shp')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiBusStops')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiSubwayStations.shp')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiSubwayStations')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'buildingCentroids.shp')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','buildingCentroids')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiCulture.shp',encoding='iso8859-1')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiCulture')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiGovFacility.shp',encoding='iso8859-1')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiGovFacility.shp')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiHealthFacility.shp',encoding='iso8859-1')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiHealthFacility')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiSport.shp',encoding='iso8859-1')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiSport')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiEdu.shp',encoding='iso8859-1')
df_count =count_points_in_polygons(gpd_points,gpd_fishnet,'id','poiEdu')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
gpd_points = gpd.read_file(poi_path+'poiParking.shp')
df_count =count_number_in_polygons(gpd_points,gpd_fishnet,'id','总泊位','poiParking')
df_fishnet = pd.merge(df_fishnet,df_count,how='left',on='fishid')
print('poi_finish')
df_fishnet = df_fishnet.rename(columns={'fishid':'id'})
df_fishnet = df_fishnet.rename(columns={'id': area_id})
df_fishnet.to_csv(middle_data_floder+'base_poi.csv',index=False)

