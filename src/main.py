# This is a sample Python script.

from src.utils import OsmUtils


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
#     rangeFile = "/Users/convel/Desktop/wenchongDst.geojson"
#     # x, y, m, n = [113.43796, 23.08660, 113.51140, 23.13701]
#     # OsmUtils.downloadOsmByBoundary(x,y,m,n,"/Users/convel/Desktop/wenchongDst.osm")
#     OsmUtils.filter_osm_by_polygon("/Users/convel/Desktop/wenchongDst.osm", rangeFile, "/Users/convel/Desktop/wenchongDst_cutted.osm")
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# # Example usage:
import pandas as pd

# 读取 Parquet 文件
# df = pd.read_parquet("/Users/convel/Desktop/广东.parquet", engine='pyarrow')
# tmp_df = df[['企业名称', '经度', '纬度']]#.to_csv('/Users/convel/Desktop/testGoutput.txt', sep=',', index=False)
# tmp_df.loc[
#     # (df['经度'].between(113.30927, 113.34742)) &              # 113.30927,23.11130 : 113.34742,23.13376
#     # (df['纬度'].between(23.11130, 23.13376))
#     (df['所属城市']=='广州市'and df['所属区县']=='天河区')
#     .to_csv('/Users/convel/Desktop/testGoutput.txt', sep=',', index=False) # `salary` 在 [3000, 8000] 范围内
# ]
# 查看数据
# print(df.head())
import re

def split_ignore_quotes(s):
    # This regular expression splits by commas that are not inside quotes
    return re.findall(r'[^,"]+|"(.*?)"', s)
gz_data = []

gz_file = open('/Users/convel/Desktop/testGZ.txt', 'w')
with open('/Users/convel/Desktop/testGoutput.txt', 'r') as f:
    gz_file.write(f.readline())
    for line in f.readlines():
        print(line)
        tmp = split_ignore_quotes(line)
        if 113.30927 <= float(tmp[1]) <= 113.34742 and 23.11130 <= float(tmp[2]) <= 23.13376:
            gz_file.write(line)
    f.close()
gz_file.flush()
gz_file.close()