import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon

area_id = 'osmid'
df_od = pd.read_csv('../data/202305广州通勤数据V2/广州通勤表_202305_rndCoord.csv')
df_od ['s_id'] = df_od.index
df_od.columns = ['source_xy','traget_grid','traget_xy','grid_len','odid','uv','type','ox','oy','dx','dy','source_grid']
df_source_grid = df_od[['source_grid','source_xy']].drop_duplicates()
df_od = []
df_source_grid.columns = ['grid','xy']
df_source_grid['list_xy'] =df_source_grid.apply(lambda z:z.xy.split(';'),axis=1)
df_source_grid['x'] = df_source_grid.apply(lambda z:float(z.list_xy[0]),axis=1)
df_source_grid['y'] = df_source_grid.apply(lambda z:float(z.list_xy[1]),axis=1)
df_source_grid['geometry'] = df_source_grid.apply(lambda z:Point(z.x,z.y),axis=1)
gpd_source_grid = df_source_grid[['grid','geometry']]
gpd_source_grid =  gpd.GeoDataFrame(gpd_source_grid,geometry='geometry', crs='epsg:4326')
gpd_source_grid = gpd_source_grid.to_crs('EPSG:4526')
gpd_source_grid.to_file('../data/base_data/od_grid.shp', driver='ESRI Shapefile', encoding='utf-8')
gpd_source_grid = []

gpd_grid = gpd.read_file('../data/base_data/od_grid.shp')

gpd_fishnet = gpd.read_file('../data/base_data/voronoi_gz.shp')
gpd_fishnet = gpd_fishnet.rename(columns={area_id: 'id'})
gpd_fishnet = gpd_fishnet[['id','geometry']]

df_join = gpd.sjoin(gpd_grid, gpd_fishnet, how='left', predicate='within')
df_join = df_join[['grid','id']]


df_od_info = pd.read_csv('../data/202305广州通勤数据V2/广州通勤明细表V2_202305.txt')
df_od_info = df_od_info[['odid','go_time','car_uv']]
df_od_info = df_od_info[~df_od_info['car_uv'].isna()]
df_od_info = df_od_info[(df_od_info['go_time'].isna())|((df_od_info['go_time']<931)&(df_od_info['go_time']>730))]
df_od_info = df_od_info.reset_index(drop=True)
df_od_info['source_grid'] = df_od_info.apply(lambda z: z.odid.split('_')[0]+'_'+z.odid.split('_')[1],axis=1)
df_od_info['target_grid'] = df_od_info.apply(lambda z: z.odid.split('_')[2]+'_'+z.odid.split('_')[3],axis=1)
df_join.columns = ['source_grid','source_id']
df_od_info = pd.merge(df_od_info,df_join,how='left')
df_join.columns = ['target_grid','target_id']
df_od_info = pd.merge(df_od_info,df_join,how='left')

df_od = df_od_info.groupby(['source_id','target_id'])['car_uv'].sum().reset_index()
df_od.to_csv('../data/base_data/base_od.csv',index=False)

