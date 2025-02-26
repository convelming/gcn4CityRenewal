import json
import os
import osmium
from shapely.geometry import Point, Polygon
import geopandas as gpd
import xml.etree.ElementTree as ET


def downloadOsmByBoundary(minLon, minLat, maxLon, maxLat, save_file,
                          URL="https://api.openstreetmap.org/api/0.6/map?bbox="):
    """

    :param minLon: WGS84
    :param minLat:
    :param maxLon:
    :param maxLat:
    :param save_file: file name and path
    :param URL: osm url prefix
    :return: null,
    """
    # # DOWNLOAD_URL="https://api.openstreetmap.org/api/0.6/map?bbox=113.1914,23.0041,113.3413,23.0909"
    # DOWNLOAD_URL = "http://overpass.openstreetmap.ru/cgi/xapi_meta?*[112.885,22.473,114.125,23.977]"
    # url_de = "https://overpass-api.de/api/map?bbox=112.885,22.473,114.125,23.977"
    # 定义下载 URL（示例使用 Geofabrik）
    osm_url = URL + f"{minLon},{minLat},{maxLon},{maxLat}"
    region = "guangzhou"  # 你可以换成其他区域
    # 使用 wget 下载 OSM 数据
    os.system(f"wget -O {save_file} {osm_url}")

def geojsonPoly2Osm(geojson_file, output_file):
    pass

