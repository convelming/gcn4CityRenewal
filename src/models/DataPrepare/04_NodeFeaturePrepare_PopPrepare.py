
import geopandas as gpd
import pandas as pd

if __name__ == '__main__':
    #文件路径
    base_data_floder = './src/models/base_data/'
    middle_data_floder = './src/models/middle_data/'
    # 基础区域和人口区域所需要用到的ID列和属性列名
    area_id_col = 'osmid'
    data_id_col,data_feature_col = 'id','a_总人'
    # 加载基础区域
    gpd_base_area = gpd.read_file(middle_data_floder+'voronoi_gz.shp') #加载基础区域
    gpd_base_area['area'] = gpd_base_area.area #各基础区域面积
    gpd_base_area['centroid'] = gpd_base_area.centroid #各基础区域中心
    #生成基础中心点的geo数据
    gpd_base_area_point = gpd_base_area[[area_id_col,'centroid','area']]
    gpd_base_area_point = gpd_base_area_point.rename(columns={'centroid':'geometry'})
    gpd_base_area_point = gpd.GeoDataFrame(gpd_base_area_point, geometry=gpd_base_area_point.geometry)
    #加载人口数据
    gpd_pop = gpd.read_file(base_data_floder+'population.shp')#面数据
    gpd_pop = gpd_pop[[data_id_col,data_feature_col,'geometry']]
    gpd_base_area_point = gpd_base_area_point.to_crs(gpd_pop.crs)
    #数据匹配
    df_join = gpd.sjoin(gpd_base_area_point, gpd_pop, how='left',  predicate='within')#基础中心点都落入哪些人口面中
    df_point = df_join[['area',area_id_col,data_id_col,data_feature_col]]#所有的基础区域中心点
    df_point_match = df_point[~df_point[data_id_col].isna()] #匹配上的中心点
    df_point_nomatch = df_point[df_point[data_id_col].isna()] #没匹配上的中心点
    df_area_nomatch = gpd_base_area[gpd_base_area[area_id_col].isin(list(df_point_nomatch[area_id_col]))].drop_duplicates() #中心点没匹配上的面数据
    df_area_nomatch = gpd.overlay(df_area_nomatch, gpd_pop, how='intersection').sort_values(by=[area_id_col, 'area'], ascending=False)[
        ['area',area_id_col,data_id_col,data_feature_col]].drop_duplicates(subset=[area_id_col]) #通过面相交搜索基础区域属于哪个人口区,使其赋上人口数据
    df_rest_nomatch = df_point_nomatch[~df_point_nomatch[area_id_col].isin(df_area_nomatch[area_id_col])] #面相交也匹配不上的基础区域
    df_point = pd.concat([df_area_nomatch, df_point_match])#合并中心点能匹配上和面相交能匹配上的数据
    #通过各个基础区域面积占比计算人口
    df_all = pd.DataFrame()
    for i in list(set(df_point[data_id_col])):#从每个人口面ID分析
        df_a = df_point[df_point[data_id_col] == i]
        area = df_a['area'].sum() #此人口面中的基础区域面积总和
        df_a['rate'] = df_a['area'] / area #各个基础区域面积占比
        df_all = pd.concat([df_all, df_a])
    df_all[data_feature_col] = df_all[data_feature_col]*df_all['rate'] #人口×面积占比表示基础区域的人口
    df_all = df_all[[area_id_col,data_feature_col]]

    #加载所有的路网点（含高速点）/基础区域的泰森多边形不含高速点
    gdf_node = gpd.read_file(middle_data_floder + 'guangzhou_drive.gpkg', layer='nodes')
    df_all = pd.merge(gdf_node[[area_id_col]], df_all, on=[area_id_col], how='left')
    df_all = df_all.fillna(0.00001)#人口赋一个极小值，确保该区域的features向量不为全0
    df_all.to_csv(middle_data_floder+'base_pop.csv',index=False)



