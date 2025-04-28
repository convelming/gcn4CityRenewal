# encoding: utf-8
import math

#google、高德经纬度转行列号[1-18]
def gaodedeg2num(lon_left,lat_down,lon_right,lat_up,zoom):
    n = 2.0 ** zoom
    #计算左下角行列号
    xtile1 = int(((lon_left + 180.0) / 360) * n)
    lat_rad1 = math.radians(lat_down)
    ytile1 = int((1.0-math.log(math.tan(lat_rad1)+(1/math.cos(lat_rad1)))/math.pi)/2.0*n)
    #计算右上角行列号
    xtile2 = int(((lon_right + 180.0) / 360) * n)
    lat_rad2 = math.radians(lat_up)
    ytile2 = int((1.0-math.log(math.tan(lat_rad2)+(1/math.cos(lat_rad2)))/math.pi)/2.0*n)
    return (xtile1,xtile2,ytile2,ytile1)

def gaodenum2deg(xtile,ytile,zoom,pix_x,pix_y):
    n=2.0 ** (1-zoom)
    lon=(n*(xtile+pix_x/256)-1)*180
    lat=(360*math.atan(math.exp(((1-n*(ytile+pix_y/256))*math.pi))))/math.pi-90
    return (lon,lat)