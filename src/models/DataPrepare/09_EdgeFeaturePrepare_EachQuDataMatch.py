import multiprocessing
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
def color_judge(r,g,b):
    if ((r>=180)&(g<=60)&(b<=60)):
        color = 3
    elif ((r>=90)&(g<=60)&(b<=60)):
        color = 4       
    elif (((r>=230)&(g>=170)&(b<=120))|((r>=220)&(g>=180)&(b>=120)&(b<=180))):
        color = 2
    else:
        color = 1
    return(color)


#周边像素点状态判断
shibie_kuandu = 2#上下左右2个栅格
def status_judge(x_loc,y_loc,shibie_kuandu,x_max,y_max,photo):
    status = 1
    for i in range(int(y_loc-shibie_kuandu),int(y_loc+shibie_kuandu+1)):
        if i >= y_max:
            break
        for j in range(int(x_loc-shibie_kuandu),int(x_loc+shibie_kuandu+1)):
            if j >= x_max:
                break    
            r,g,b = photo[i,j]
            c = color_judge(r,g,b)
            if c ==4:
                status = 4
                break
            if c > status:
                status = c
        if status ==4:
            break
    return(status) 

#路段点删除或保留判断
def retain_judge(df,year,month,day,hour,minute):
    df_du = df[(df[str(year)+str(month)+str(day)+str(hour)+str(minute)]>=2)]
    list_du_orig = list(set(df_du['ORIG_FID']))        
    df_retain = pd.DataFrame()
    for orig in list_du_orig:
        df_orig = df_du[df_du['ORIG_FID']==orig]
        orig_start_id = df[df['ORIG_FID']==orig]['FID'].min()
        orig_end_id = df[df['ORIG_FID']==orig]['FID'].max()
        start_id = df_orig['FID'].min()
        end_id = df_orig['FID'].max()
        break_num = end_id - start_id - len(df_orig) + 1
        if (break_num==0):#连续的
            if len(df_orig)>4:#连续且数量大
                df_retain = pd.concat([df_retain,df_orig])#保留
        else:#存在不连续
            df_cha = df_orig[['FID',str(year)+str(month)+str(day)+str(hour)+str(minute)]].reset_index(drop = True)
            df_cha['idx'] = df_cha.index
            df_cha2 = df_cha[1:].reset_index(drop = True)
            df_cha2['idx'] = df_cha2.index
            df_cha = pd.merge(df_cha[0:-1],df_cha2,on = 'idx')#序列错位交叉
            df_cha['cha'] = df_cha['FID_y'] - df_cha['FID_x']#计算前后序列差
            df_nocha = df_cha[df_cha['cha']==1]#连续点的序列
            df_lian = df_du[df_du['FID'].isin(list(df_nocha['FID_x'])+list(df_nocha['FID_y'])+list(df_cha['FID_x'])+list(df_cha['FID_y']))]#连续点的状态数据
            df_cha = df_cha[df_cha['cha']>1].sort_values(by='FID_x').reset_index(drop = True)#不连续点的起终点
            for i in range(0,len(df_cha)):
                #补全
                if df_cha['cha'][i] <6 :#断开长度小于6时补全
                    df_fill = df[(df['FID']>df_cha['FID_x'][i])&(df['FID']<df_cha['FID_y'][i])]#补全从不连续起点到终点
                    df_fill[str(year)+str(month)+str(day)+str(hour)+str(minute)] = df_cha[str(year)+str(month)+str(day)+str(hour)+str(minute)+'_x'][i]#补全的状态为上游的状态
                    if i == 0 :#是第一个断点
                        if i == (len(df_cha)-1):#同时也是最后一个断点
                                if (end_id-start_id)>=5 :#补全后为整段路
                                    df_lian = pd.concat([df_lian,df_fill])#存储补全的数据
                                else:
                                    df_lian = df_lian[(df_lian['FID']>df_cha['FID_x'][i])]#剔除路段中间连续数量少的点     
                        else:#不是最后一个断点，
                            if (df_cha['FID_x'][i+1] - start_id)>=5 :#下一个断点的起点减路段起点
                                df_lian = pd.concat([df_lian,df_fill])#补全的数据
                            else:
                                df_lian = df_lian[(df_lian['FID']>df_cha['FID_x'][i])]#剔除路段中间连续数量少的点                                  
                    else:#不是第一个断点
                        if i == (len(df_cha)-1):#是最后一个断点
                                if (end_id - df_cha['FID_y'][i-1])>=5 :#终点减上一个断点的终点
                                    df_lian = pd.concat([df_lian,df_fill])#补全的数据  
                                else:
                                    df_lian = df_lian[(df_lian['FID']>df_cha['FID_x'][i])|(df_lian['FID']<df_cha['FID_y'][i-1])]#剔除路段中间连续数量少的点                                                       
                        else:#同时也不是最后一个断点
                                if (df_cha['FID_x'][i+1] - df_cha['FID_y'][i-1])>=5 :#下一个断点的起点减上一个断点的终点
                                    df_lian = pd.concat([df_lian,df_fill])#补全的数据                              
                                else:
                                    df_lian = df_lian[(df_lian['FID']>df_cha['FID_x'][i])|(df_lian['FID']<df_cha['FID_y'][i-1])]#剔除路段中间连续数量少的点                   
                #剔除    
                else:
                    if i == 0 :#第一个断点
                        if i == (len(df_cha)-1):
                            if (end_id - df_cha['FID_y'][i])< 5:#连续的长度小于6
                                df_lian = df_lian[df_lian['FID']<df_cha['FID_y'][i]]#剔除出口道连续数量少的点                        
                        if (df_cha['FID_x'][i] - start_id )< 5 :#连续的长度小于6
                            df_lian = df_lian[df_lian['FID']>df_cha['FID_x'][i]]#剔除进口道连续数量少的点
                    else:
                        if (df_cha['FID_x'][i] - df_cha['FID_y'][i-1] )< 5 :#连续的长度小于5
                            df_lian = df_lian[(df_lian['FID']>df_cha['FID_x'][i])|(df_lian['FID']<df_cha['FID_y'][i-1])]#剔除路段中间连续数量少的点
                        if i == (len(df_cha)-1):
                            if (end_id - df_cha['FID_y'][i])< 5:#连续的长度小于6
                                df_lian = df_lian[df_lian['FID']<df_cha['FID_y'][i]]#剔除出口道连续数量少的点               
            df_retain = pd.concat([df_retain,df_lian])
    return(df_retain)

#当前时刻状态获取
def status_add(img,year,month,day,hour,minute,df_copy):
    df_sta = df_copy.copy()
    df_sta[str(year)+str(month)+str(day)+str(hour)+str(minute)] = df_sta.apply(lambda z:status_judge(z.x_png,z.y_png,shibie_kuandu,w,h,img),axis=1)
    df_yongdu = df_sta
#     df_yongdu = retain_judge(df_sta,year,month,day,hour,minute)
#     print(df_yongdu)
    df_noyongdu = df_sta[~df_sta['FID'].isin(list(df_yongdu['FID']))]
    df_noyongdu[str(year)+str(month)+str(day)+str(hour)+str(minute)] = 1
    df_sta = pd.concat([df_yongdu,df_noyongdu])
    df_sta = df_sta[['FID',str(year)+str(month)+str(day)+str(hour)+str(minute)]]
    return(df_sta)

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

