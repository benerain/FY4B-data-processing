嗯，我现在需要将FY4B卫星数据和地基观测数据融合成一个适合AI训练的数据集，使用Zarr格式存储。首先，我需要用satpy处理卫星数据，然后裁剪到中国区域，最后整理成Zarr。用户提供了一个样例代码，我需要根据这个样例来制作一个Zarr的示例。

我的项目最终目的是将凝视卫星FY4B数据融合165个24h地基观测数据，整理成 ready for AI training的数据集。首先需要通过satpy工具对通道数据进行处理和定标（利用内部的功能），然后需要裁剪中国区域，然后要整理成zarr的格式，请梳理上述技术思路，

我有一个样例L1通道数据，请告诉我

filenames = glob.glob('/Users/liyujia/Projects/FY4B-data-processing/data/raw/L1/4000M/FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_*_4000M_V*.HDF')
# 创建scene对象
scn = Scene(filenames,reader='agri_fy4b_l1')

# 查看可用的通道
scn.available_dataset_names()

['C01',
 'C02',
 'C03',
 'C04',
 'C05',
 'C06',
 'C07',
 'C08',
 'C09',
 'C10',
 'C11',
 'C12',
 'C13',
 'C14',
 'C15']

 请问我如何尝试制作zarr 样例





from satpy import Scene
from satpy.readers import load_reader_configs

# 强制重新加载配置
load_reader_configs(["/path/to/satpy/etc/readers"], clear_cache=True)

# 检查所有可用读取器
print(Scene.available_readers())


from satpy import Scene

clm_file_path = '/Users/liyujia/Projects/FY4B-data-processing/data/raw/L2/FY4B-_AGRI--_N_DISK_1050E_L2-_CLM-_MULT_NOM_20250226070000_20250226071459_4000M_V0001.NC'

scn2 = Scene(reader = 'agri_fy4b_l2' , filenames = [clm_file_path])
scn2.load(['cloud_mask'])
print(scn2.available_dataset_names())