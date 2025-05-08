import geopandas as gpd
import pandas as pd
import osmnx as ox
import networkx as nx

if __name__ == '__main__':
    list_qu_name = ['th','yx','lw','by','hz','py','hp','ns','hd','zc','ch']
    list_central = ['th','yx','lw','by','hz','py']
    middle_data_floder = './src/models/middle_data/'
    congestion_result_file_path = middle_data_floder+'congestion_edge/congestion_result/'
    gr = ox.load_graphml(middle_data_floder+'guangzhou_drive_feature.graphml')
    df_congestion_result = pd.read_csv(congestion_result_file_path+'all_congestion.csv')

    gpd_qu_all = pd.DataFrame()
    for qu_name in list_qu_name:
        gpd_qu = gpd.read_file(congestion_result_file_path+ 'shp/congestion_result_'+qu_name+'.shp')
        gpd_qu_all = pd.concat([gpd_qu_all,gpd_qu])

    for i in range(len(gpd_qu_all)):
        if i%1000 == 0:
            print(i,len(gpd_qu_all))
        nx.set_edge_attributes(gr, {(gpd_qu_all['u'][i],gpd_qu_all['v'][i],0): {'volumes':list(gpd_qu_all[[str(hour) for hour in range(0,24)]][i:i+1].values[0])}}  )
        nx.set_edge_attributes(gr, {(gpd_qu_all['u'][i],gpd_qu_all['v'][i],1): {'volumes':list(gpd_qu_all[[str(hour) for hour in range(0,24)]][i:i+1].values[0])}}  )
        nx.set_edge_attributes(gr, {(gpd_qu_all['u'][i],gpd_qu_all['v'][i],2): {'volumes':list(gpd_qu_all[[str(hour) for hour in range(0,24)]][i:i+1].values[0])}}  )

    ox.save_graph_geopackage(gr, middle_data_floder+'guangzhou_drive_feature_node&edge.gpkg')
    ox.save_graphml(gr, middle_data_floder+'guangzhou_drive_feature_node&edge.graphml')