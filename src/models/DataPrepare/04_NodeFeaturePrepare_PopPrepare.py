import warnings

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore")
poi_path = './src/models/base_data/'
middle_data_floder = './src/models/middle_data/'
area_id = 'osmid'

gpd_fishnet = gpd.read_file(middle_data_floder+'voronoi_gz.shp')

gpd_fishnet['area'] = gpd_fishnet.area
gpd_fishnet = gpd_fishnet.rename(columns={area_id: 'id'})

gpd_fishnet['centroid'] = gpd_fishnet.centroid
gpd_fishpoint = gpd_fishnet[['id','centroid','area']]
gpd_fishpoint = gpd_fishpoint.rename(columns={'centroid':'geometry'})
gpd_fishpoint =  gpd.GeoDataFrame(gpd_fishpoint, geometry=gpd_fishpoint.geometry)

gpd_pop = gpd.read_file(poi_path+'population.shp')
gpd_pop = gpd_pop[['id','a_总人','geometry']]
gpd_fishpoint = gpd_fishpoint.to_crs(gpd_pop.crs)
gpd_fishpoint = gpd_fishpoint.rename(columns={'id': 'pointid'})

df_join = gpd.sjoin(gpd_fishpoint, gpd_pop, how='left',  predicate='within')
df_area = df_join[['pointid','area','id','a_总人']]
df_area_match = df_area[~df_area['id'].isna()]
df_area_nomatch = df_area[df_area['id'].isna()]

df_need = gpd_fishnet[gpd_fishnet['id'].isin(list(df_area_nomatch['pointid']))].drop_duplicates()
df_need = df_need.rename(columns={'id':'pointid'})
df_area_nomatch = gpd.overlay(df_need,gpd_pop,how='intersection').sort_values(by=['pointid','area'],ascending=False)[['pointid','id','area','a_总人']].drop_duplicates(subset=['pointid'])
df_area = pd.concat([df_area_nomatch,df_area_match])
df_all = pd.DataFrame()
for i in list(set(df_area['id'])):
    df_a = df_area[df_area['id']==i]
    area = df_a['area'].sum()
    df_a['rate'] = df_a['area']/area
    df_all = pd.concat([df_all,df_a])
df_area = pd.merge(df_area,df_all,on=['pointid','id','area','a_总人'])
df_area['a_总人'] = df_area['a_总人']*df_area['rate']

df_area= df_area[['pointid','a_总人']]
df_area = df_area.rename(columns={'pointid': 'id'})
df_area = df_area.rename(columns={'id': area_id})
df_area.to_csv(middle_data_floder+'base_pop.csv',index=False)

