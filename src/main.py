# This is a sample Python script.
from src.utils import OsmUtils


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    rangeFile = "/Users/convel/Desktop/wenchongDst.geojson"
    # x, y, m, n = [113.43796, 23.08660, 113.51140, 23.13701]
    # OsmUtils.downloadOsmByBoundary(x,y,m,n,"/Users/convel/Desktop/wenchongDst.osm")
    OsmUtils.filter_osm_by_polygon("/Users/convel/Desktop/wenchongDst.osm", rangeFile, "/Users/convel/Desktop/wenchongDst_cutted.osm")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Example usage:
