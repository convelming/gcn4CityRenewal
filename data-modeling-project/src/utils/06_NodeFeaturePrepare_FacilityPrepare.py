import multiprocessing
import warnings
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import pandas as pd

area_id = 'osmid'
poi_path = '../data/base_gis/'
gpd_facility = gpd.read_file(poi_path+'Guangzhou_Buildings_DWG-Polygon4526.shp')
gpd_facility['fid'] = gpd_facility.index
gpd_facility['cen'] = gpd_facility.apply(lambda z:z.geometry.centroid,axis=1)
gpd_fpoint = gpd_facility[['fid','cen']]
gpd_fpoint = gpd.GeoDataFrame(gpd_fpoint, geometry=gpd_fpoint.cen)
gpd_fishnet = gpd.read_file('../data/base_data/voronoi_gz.shp')
df_join = gpd.sjoin(gpd_fpoint, gpd_fishnet, how='left', predicate='within')
df_join =df_join[['fid',area_id]]
df_join = pd.merge(df_join,gpd_facility)
df_join['all_area'] = df_join.apply(lambda z:z.Elevation/3*z.area,axis=1)

df_gb_area = df_join.groupby(area_id)['area'].sum().reset_index()
df_gb_all_area = df_join.groupby(area_id)['all_area'].sum().reset_index()
df_join.to_csv('../data/base_data/facility/facility_raw.csv',index=False)
df_gb = pd.merge(df_gb_area,df_gb_all_area)
df_gb.to_csv('../data/base_data/base_facility_area.csv',index=False)