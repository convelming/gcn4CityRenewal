import osmnx as ox
import geopandas as gpd

middle_data_floder = './src/models/middle_data/'
#下载广州OSM路网
# G = ox.graph_from_place('guangzhou', network_type="drive")
# ox.save_graphml(G, "./base_data/guangzhou_drive.graphml")
# ox.save_graph_geopackage(G, "./base_data/guangzhou_drive.gpkg")
#读取已下载好的广州OSM路网
G = ox.load_graphml(middle_data_floder+'guangzhou_drive.graphml')
gdf_node = gpd.read_file(middle_data_floder+'guangzhou_drive.gpkg',layer='nodes')
gdf_edge = gpd.read_file(middle_data_floder+'guangzhou_drive.gpkg',layer='edges')
#将高速公路的节点移除
df_motor = gdf_edge[gdf_edge['highway'].str.contains('motor')][['u','v']]
list_motor = list(set(list(df_motor['u'])+list(df_motor['v'])))
gdf_node_nomotor = gdf_node[~gdf_node['osmid'].isin(list_motor)]
gdf_node_nomotor.to_file(middle_data_floder+'guangzhou_drive_nomotor.gpkg')

# 在QGIS中生成不含高速公路节点的泰森多边形(QGIS对guangzhou_drive_nomotor.gpkg文件进行操作，得到./middle_data/voronoi_gz.shp)
# （代码02-08）通过泰森多边形和POI/landuse/POP/BUildings/RoadLength/Facility/OD等进行空间匹配，获得各个泰森多边形的features，并附到节点上.
# （代码09-11）通过QGIS将广州路网分为各个区的shp文件，在QGIS中空间调整使其与爬取的路况图片位置匹配，得到./middle_data/congestion_edge/osm_edge_qu/中的各区路网。
# （代码09-11）通过代码09将路况颜色赋值到各区路网的边上，并计算24小时的路况均值，得到./middle_data/congestion_edge/congestion_result/中的匹配结果；并通过代码10生成shp结果
# （代码09-11）通过代码11将路况结果赋值到graphml文件上。
# （代码12-14）通过代码12生成不经过GCN卷积的变区域周边节点属性的输入特征训练集，结构为[Input_x,flowcounts],训练集的数量可增加，20万的训练集预测效果提升已达峰。
# （代码12-14）通过代码13生成经过GCN卷积的后的各个节点属性以及GCN权重，存于middle_data/gcn_features/中。
# （代码12-14）通过代码14生成经过GCN卷积的后的变区域周边节点属性的输入特征训练集，生成从1-20层卷积各20万训练集，存于TrainData/中。