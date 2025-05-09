import ast
import random
import geopandas as gpd

import osmnx as ox
import pandas as pd
import math
import pickle

from sklearn.cluster import KMeans
#点Features提取
def count_points_in_polygons(gpd_points,gpd_polygons,polygon_col,new_col_name):
    gpd_points = gpd_points[['geometry']].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = pd.DataFrame()
    df_vc[polygon_col] = df_join[polygon_col].value_counts().index
    df_vc[new_col_name] = df_join[polygon_col].value_counts().values
    return(df_vc)

def count_number_in_polygons(gpd_points,gpd_polygons,polygon_col,cal_num_col,new_col_name):
    gpd_points = gpd_points[['geometry',cal_num_col]].drop_duplicates()
    gpd_points = gpd_points.to_crs(gpd_polygons.crs)
    df_join = gpd.sjoin(gpd_points, gpd_polygons, how='left', predicate='within')
    df_vc = df_join.groupby([polygon_col])[cal_num_col].sum().reset_index()
    df_vc.columns = [polygon_col,new_col_name]
    return(df_vc)


#边流量提取
import os
import geopandas as gpd
import pandas as pd
from Lonlat_Tile import gaodedeg2num,gaodenum2deg
from coordTransform import gcj02_to_wgs84

from PIL import ImageFile
from PIL import Image
from skimage import io,data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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

