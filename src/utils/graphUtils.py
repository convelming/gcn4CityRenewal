



###
'''
Looking for similar region is not an easy task

**提取子图**
import networkx as nx
target_subgraph = G.subgraph(target_subgraph_nodes)
'''
如果目标子图是一个**多边形范围**（Shapefile 或 GeoJSON），可以用 `geopandas` 进行空间过滤：
```python
import geopandas as gpd
from shapely.geometry import Point

gdf_nodes = gpd.GeoDataFrame(G.nodes(data=True), geometry=[Point(d['x'], d['y']) for _, d in G.nodes(data=True)])
polygon = gpd.read_file('target_area.geojson').geometry.iloc[0]
target_subgraph_nodes = gdf_nodes[gdf_nodes.geometry.within(polygon)].index
target_subgraph = G.subgraph(target_subgraph_nodes)


def getSubGraph(gdf_nodes,poly_points):
