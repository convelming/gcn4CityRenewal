import geopandas as gpd
import pandas as pd

list_qu_name = ['th','yx','lw','by','hz','py','hp','ns','hd','zc']
list_central = ['th','yx','lw','by','hz','py']
congestion_edge_file_path = '../data/congestion_edge/osm_edge_qu/'
congestion_result_file_path = '../data/congestion_edge/congestion_result/'

df_all = pd.DataFrame()
for qu_name in list_qu_name:
    df_qu_result = pd.read_csv(congestion_result_file_path + qu_name + '/'+str(int(0))+'.csv')
    df_qu_result.columns = ['fid',str(int(0))]    
    for hour_num in range(1,24):
        qu_hour_result_path = congestion_result_file_path + qu_name + '/'+str(int(hour_num))+'.csv'        
        df_result = pd.read_csv(qu_hour_result_path)
        df_result.columns = ['fid',str(int(hour_num))]        
        df_qu_result = pd.merge(df_qu_result,df_result)
    df_all = pd.concat([df_all,df_qu_result]) 
df_all = df_all.drop_duplicates()
df_all = df_all.sort_values(by='fid').reset_index(drop=True)  

def consolidate_dataframe(df):
    # 定义需要求均值的列（B0到B23）
    cols_to_mean = [f'{i}' for i in range(24)]
    
    # 按A列分组并聚合
    result = df.groupby('fid', as_index=False).agg({
        **{col: 'mean' for col in cols_to_mean},
    })
    return result 

consolidated_df = consolidate_dataframe(df_all)    

if not os.path.exists(congestion_result_file_path+'shp/'):
    os.makedirs(congestion_result_file_path+'shp/')
for qu_name in list_qu_name:
    gpd_osm_edge = gpd.read_file(congestion_edge_file_path+'osm_'+qu_name+'.shp')    
    df_congestion_qu = pd.merge(gpd_osm_edge,consolidated_df)
    df_congestion_qu.to_file(congestion_result_file_path+ 'shp/congestion_result_'+qu_name+'.shp')