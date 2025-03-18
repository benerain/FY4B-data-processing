import xarray as xr
import datetime as dt
import logging
from satpy.readers.file_handlers import BaseFileHandler
from pyresample.geometry import AreaDefinition

logger = logging.getLogger(__name__)
    
RESOLUTION_LIST = [250, 500, 1000, 2000, 4000]


from satpy.readers.netcdf_utils import NetCDF4FileHandler  # 改为继承 NetCDF 处理器
from satpy.readers.fy4_base import FY4Base  # 假设 FY4Base 在 fy4_base.py 中
from satpy.readers._geos_area import get_area_extent, get_area_definition

class AGRI_L2_NC_Handler( NetCDF4FileHandler):  
    def __init__(self, filename, filename_info, filetype_info):

        super().__init__(filename, filename_info, filetype_info)

        self.nc = xr.open_dataset(filename, decode_cf=False)

        self.sensor = "AGRI"  # 强制指定传感器类型


        # info of 250m, 500m, 1km, 2km and 4km data
        self._COFF_list = [21983.5, 10991.5, 5495.5, 2747.5, 1373.5]
        self._LOFF_list = [21983.5, 10991.5, 5495.5, 2747.5, 1373.5]

        self._CFAC_list = [163730199.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]
        self._LFAC_list = [163730199.0, 81865099.0, 40932549.0, 20466274.0, 10233137.0]


        self.subpoint_lon = float(filename_info['longitude'][:-1])/10
        if filename_info['longitude'][-1] == 'W':
            self.subpoint_lon *= -1
        self.subpoint_lat = 0.0

        # self.satellite_height = 35786000.0 * 1000
        # self.satellite_height = float(self.nc["nominal_satellite_height"].values[0]) * 1000

        if "nominal_satellite_height" in self.nc.variables:
            self.satellite_height = float(self.nc["nominal_satellite_height"].values[0]) 
        else:
            # 若文件缺失，则可用 35786000.0 这个经典 GEO 高度
            self.satellite_height = 35786000.0

        # 正确：使用文件中声明的GRS80标准
        # self.a = 6378137.0  # 长半轴（WGS84）
        # self.b = 6356752.3  # 短半轴（WGS84）
        self.a = 6378137.0   # GRS80长半轴
        self.b = 6356752.3   # GRS80短半轴

        # 固定参数（根据文件维度）
        self.ncols = self.nc.dims["x"]
        self.nlines = self.nc.dims["y"]


        self._start_time = filename_info.get("start_time", None)
        self._end_time = filename_info.get("end_time", None)


    @property
    def start_time(self):
        """数据起始时间"""
        return self._start_time

    @property
    def end_time(self):
        """数据结束时间"""
        return self._end_time


    def get_dataset(self, dataset_id, dataset_info):
        """读取数据集（例如云掩膜）"""
        print(f"🟢 读取数据集: {dataset_id}, info: {dataset_info}")
        if dataset_id["name"] == "CLM":
            data = self.nc["CLM"]
            # data = data.transpose ( "y" , "x" ) # 有必要吗 将维度顺序从 (x, y) 改为 (y, x), 因为satpy默认期望数据维度顺序为 (y, x)

            # data = data.rename({'x': 'y', 'y': 'x'}) # 不知道是哪个

            if data.ndim >= 2:
                data = data.rename({data.dims[-2]: "y", data.dims[-1]: "x"})

            data.attrs.update({
                "units": "1",
                "standard_name": "Cloud_Mask",
                "platform_name": "FY-4B",
                "sensor": "AGRI",
                "fill_value": 127,
                "valid_range": (0, 126),
                'area': self.get_area_def(dataset_id) 
            })
            return data
        

    def get_area_def(self, dataset_id): #  很重要
        """生成 AreaDefinition"""
        # res = int(self.filename_info[ "resolution"])
        res = dataset_id.get("resolution", 4000)
        pdict = { }
        # 从文件变量获取行列起始位置（假设L2数据是全盘，起始位置为0）

        # 这里参考 L1 里的计算方法 ( begin_cols, end_lines 等视情况而定 )
        # 如果 L2 文件没有“Begin Pixel Number”和“End Line Number”之类属性，可固定成 0, nlines-1
        begin_cols = 0
        end_lines = self.nlines - 1

        pdict = {}
        pdict["coff"] = (self._COFF_list[RESOLUTION_LIST.index(res)]
                         - begin_cols + 1)
        pdict["loff"] = (-self._LOFF_list[RESOLUTION_LIST.index(res)]
                         + end_lines + 1)
        pdict["cfac"] = self._CFAC_list[RESOLUTION_LIST.index(res)]
        pdict["lfac"] = self._LFAC_list[RESOLUTION_LIST.index(res)]

        pdict["a"] = self.a
        pdict["b"] = self.b
        pdict["h"] = self.satellite_height
        pdict["ssp_lon"] = self.subpoint_lon

        pdict["nlines"] = self.nlines
        pdict["ncols"] = self.ncols

        # 固定值/其他元信息
        pdict["scandir"] = "N2S"
        pdict["a_desc"] = self.nc.attrs.get("Title", "FY4B L2 Data")
        pdict["a_name"] = f"{self.nc.attrs.get('scene_id','FullDisk')}_{res}m"
        pdict["p_id"] = f"{self.nc.attrs.get('platform_ID','FY4B')}, {res}m"

        # 计算投影范围
        area_extent = get_area_extent(pdict)
        # 生成 area definition
        area = get_area_definition(pdict, area_extent)
        # print(f"🟢 生成 area definition: {area}")

        return area


