import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

def merge_features(gpd_base,features_type): #合并Features
    df_features = pd.read_csv(f'{middle_data_floder}{features_type}.csv')
    gpd_base = pd.merge(gpd_base,df_features,on=[base_id_col_name],how='left')
    return(gpd_base)

if __name__ == '__main__':
    # 文件路径
    middle_data_floder = './src/models/middle_data/'
    # 加载基础路网
    gdf_node = gpd.read_file(middle_data_floder + 'guangzhou_drive.gpkg', layer='nodes')
    base_id_col_name = 'osmid'#路网节点ID
    #合并Features文件
    list_poi_type = ['base_pop','base_landuse','base_road','base_poi','base_facility_area']
    df_input = gdf_node.copy()
    for poi_type in list_poi_type:
        df_input = merge_features(df_input,poi_type)
    df_input = df_input.fillna(0)#空值赋0
    df_input.to_csv(middle_data_floder+'InputFeature.csv',index=False)#导出结果
    #将数据加载到graphml中
    list_feature_col = df_input.columns.to_list()[len(gdf_node.columns.to_list()):] #提取Features的列名
    gr = ox.load_graphml(middle_data_floder+'guangzhou_drive.graphml') #加载graphml文件
    for i in range(len(gdf_node)):#给节点添加features属性
        if i % 1000 == 0:
            print(i,len(gdf_node))
        nx.set_node_attributes(gr,{df_input[base_id_col_name][i]:{"features":list(df_input[list_feature_col][i:i+1].values[0])}} )
    ox.save_graph_geopackage(gr, middle_data_floder+'guangzhou_drive_feature.gpkg')#导出结果
    ox.save_graphml(gr, middle_data_floder+'guangzhou_drive_feature.graphml')#导出结果
