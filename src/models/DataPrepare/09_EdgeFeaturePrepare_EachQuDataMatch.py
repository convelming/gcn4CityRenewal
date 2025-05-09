import warnings
import os
import geopandas as gpd
import pandas as pd
from Lonlat_Tile import gaodedeg2num,gaodenum2deg
from coordTransform import gcj02_to_wgs84
warnings.filterwarnings("ignore")
from PIL import ImageFile
from PIL import Image
from skimage import io,data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
#颜色状态判断4颜色
middle_data_floder = './src/models/middle_data/'
from src.models.DataPrepare.DataPrepareUtils import status_add

if __name__ == '__main__':
    list_qu_name = ['th','yx','lw','by','hz','py','hp','ns','hd','zc','ch']
    list_central = ['th','yx','lw','by','hz','py']
    congestion_edge_file_path = middle_data_floder+'congestion_edge/osm_edge_qu/'
    congestion_result_file_path = middle_data_floder+'congestion_edge/congestion_result/'

    for qu_name in list_qu_name:
        for hour_num in range(24):
            qu_hour_result_path = congestion_result_file_path + qu_name + '/'+str(int(hour_num))+'.csv'
            if os.path.exists(qu_hour_result_path):
                print(qu_name,hour_num)
                continue
            else:
                congestion_data_file_path = 'G:/拥堵爬取数据/'
                # gpd_qu_edge = gpd.read_file(congestion_edge_file_path+'osm_'+qu_name+'.shp')
                if qu_name in list_central:
                    congestion_data_file_path = congestion_data_file_path + '结果-中心区-20240513-20240517/'
                    day_num = '15'
                    list_day_num = ['13','14','15','16','17']
                if qu_name == 'hp':
                    congestion_data_file_path = congestion_data_file_path + '结果-黄埔区-20240109-20240116/'
                    day_num = '10'
                    list_day_num = ['09','10','11','12','15']
                if qu_name == 'ns':
                    congestion_data_file_path = congestion_data_file_path + '结果-南沙区-20240702-20240708/'
                    day_num = '03'
                    list_day_num = ['02','03','04','05','08']
                if qu_name == 'hd':
                    congestion_data_file_path = congestion_data_file_path + '结果-花都区-20240718-20240726/'
                    day_num = '18'
                    list_day_num = ['19','22','23','24','25']
                if qu_name == 'zc':
                    congestion_data_file_path = congestion_data_file_path + '结果-增城区-20250321-20250327/'
                    day_num = '21'
                    list_day_num = ['21','24','25','26','27']
                with open(congestion_data_file_path+"01爬取_settings.txt","r") as f:
                    line=f.readlines()
                    leftjingdu=float(line[2])
                    downweidu=float(line[3])
                    rightjingdu=float(line[5])
                    upweidu=float(line[6])
                    zoom=int(line[8])
                    type=int(line[10])
                    time_interval = float(line[12])
                x_min, x_max, y_min, y_max = gaodedeg2num(leftjingdu, downweidu, rightjingdu,upweidu, zoom)  # 获取行列号，计算拼合的图片的大小
                lenx = x_max - x_min + 1
                leny = y_max - y_min + 1
                num = lenx * leny
                x1, y1 = gaodenum2deg(x_min, y_max, zoom, 0, 256)  # 左下角经纬度
                x2, y2 = gaodenum2deg(x_max, y_min, zoom, 256, 0)  # 右上角经纬度
                zuobiao1 = gcj02_to_wgs84(x1, y1)
                zuobiao2 = gcj02_to_wgs84(x2, y2)
                jingdu_cha = zuobiao2[0]-zuobiao1[0]
                weidu_cha = zuobiao2[1]-zuobiao1[1]
                gpd_osm_edge = gpd.read_file(congestion_edge_file_path+'osm_'+qu_name+'.shp')
                gpd_osm_edge = gpd_osm_edge.to_crs(epsg=4326)
                gpd_osm_edge['x'] = gpd_osm_edge.apply(lambda z:z.geometry.coords[0][0] if z.geometry.type=='LineString' else z.geometry.geoms[0].coords[0][0],axis=1)
                gpd_osm_edge['y'] = gpd_osm_edge.apply(lambda z:z.geometry.coords[0][1] if z.geometry.type=='LineString' else z.geometry.geoms[0].coords[0][1],axis=1)
                df_raw = gpd_osm_edge[['fid','x','y']]
                df_raw.columns = ['FID','x','y']
                folder_path = congestion_data_file_path+"/01基础数据/roadstatus/"
                folder_names = os.listdir(folder_path)
                df_area = df_raw[(df_raw['x']>zuobiao1[0])&(df_raw['x']<zuobiao2[0])&(df_raw['y']>zuobiao1[1])&(df_raw['y']<zuobiao2[1])]
                df_area = df_area.reset_index(drop=True)
                for n in range(0,len(folder_names)):
                    folder = folder_names[n]
                    if folder[-1]== 'g':
                        img = io.imread( folder_path +'/'+ folder )
                        h,w,t = img.shape
                        df_area['x_png'] = df_area.apply(lambda z:(z.x-zuobiao1[0])/jingdu_cha * w,axis=1)
                        df_area['y_png'] = df_area.apply(lambda z:(zuobiao2[1]-z.y)/weidu_cha * h,axis=1)
                        break
                qu_result_path = congestion_result_file_path + qu_name
                if not os.path.exists(qu_result_path):
                    # 如果文件夹不存在，则创建文件夹
                    os.makedirs(qu_result_path)
                for folder in folder_names:
                    if folder[-1]== 'g':
                        year = folder[-21:-19]
                        month = folder[-18:-16]
                        day = folder[-15:-13]
                        hour = folder[-12:-10]
                        minute = folder[-9:-7]
                        if (day in list_day_num)& (int(hour)==hour_num):
                            print(month,day,hour,minute,qu_name)
                            img = io.imread( folder_path +'/'+ folder )
                            h,w,t = img.shape
                            df_folder = status_add(img,year,month,day,hour,minute,df_area)
                            df_area = pd.merge(df_area,df_folder,on=['FID'])
                df_hour = pd.DataFrame()
                df_hour['hid'] = list(set(df_area.columns.tolist())-set(['FID','x','y','x_png','y_png']))
                df_hour['hour'] = df_hour.apply(lambda z:int(z.hid[6:8]),axis=1)
                df_hour = df_hour.sort_values(by='hour')
                df_congestion_hour = df_area[['FID']]
                list_hour = list(df_hour[df_hour['hour']==hour_num]['hid'])
                df_congestion_hour[str(hour)] = list(df_area[list_hour].mean(axis=1))
                df_congestion_hour = df_congestion_hour.rename(columns={'FID':'fid'})
                df_congestion_hour.to_csv(qu_result_path+ '/'+str(int(hour_num))+'.csv',index=False)

