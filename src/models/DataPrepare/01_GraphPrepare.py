import osmnx as ox
import geopandas as gpd
#下载广州OSM路网
# G = ox.graph_from_place('guangzhou', network_type="drive")
# ox.save_graphml(G, "./base_data/guangzhou_drive.graphml")
# ox.save_graph_geopackage(G, "./base_data/guangzhou_drive.gpkg")
#读取已下载好的广州OSM路网
G = ox.load_graphml('../data/base_data/guangzhou_drive.graphml')
gdf_node = gpd.read_file("../data/base_data/guangzhou_drive.gpkg",layer='nodes')
gdf_edge = gpd.read_file("../data/base_data/guangzhou_drive.gpkg",layer='edges')
#将高速公路的节点移除
df_motor = gdf_edge[gdf_edge['highway'].str.contains('motor')][['u','v']]
list_motor = list(set(list(df_motor['u'])+list(df_motor['v'])))
gdf_node_nomotor = gdf_node[~gdf_node['osmid'].isin(list_motor)]
gdf_node_nomotor.to_file('../data/base_data/guangzhou_drive_nomotor.gpkg')

# 生成不含高速公路节点的泰森多边形(QGIS对guangzhou_drive_nomotor.gpkg问间进行操作，得到./base_data/voronoi_gz.shp)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 通过泰森多边形和POI/landuse/POP/BUildings/RoadLength等进行空间匹配
# 获得各个泰森多边形的features，并附到节点上