import os
import glob
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from datetime import datetime
import os

import glob


from datetime import datetime


def read_nc(filename):
	"""
	Read netcdf file and return xarray.Dataset.

	:param filename: Tile filename.
	:returns: xarray.Dataset of the tile
	"""
	file = xr.open_dataset(filename)
	return file


def subscalegrid(output_size, sample):
	"""
	Subscale the input grid in a sample to a given size.

	:param output_size: Desired output size for the grid element.
	:param sample: Data array.
	:returns: Subscaled input grid element.
	"""
	assert output_size < sample.shape[1]
	size = sample.shape[1]
	if output_size % 2 == 0:
		image = sample[:,
		        size // 2 - output_size // 2: size // 2 + output_size // 2,
		        size // 2 - output_size // 2: size // 2 + output_size // 2]
	else:
		image = sample[:,
		        size // 2 - output_size // 2: size // 2 + output_size // 2 + 1,
		        size // 2 - output_size // 2: size // 2 + output_size // 2 + 1]
	return image


class FY4BTilesDataset(Dataset):
	"""
	Pytorch 数据集类，用于加载 FY 平台生成的云属性瓦片。
  
	"""

	def __init__(self, 
				 data_dir, 
				 tile_pattern, 
	
				 subset =None,  # 使用通道索引而非变量名
				 subset_cols=None,    # 可选：为通道提供语义化名称
				 transform=None, 
				 normalize=False,
				 means=None,
				 stds=None,
				 mins=None,
				 grid_size= 256,
				 log_transform=False,
				#  log_transform_indices=None
				 ):  # 指定哪些通道需要对数变换
		
		"""
		FY4B卫星数据切片的PyTorch数据集类，处理通道数据
		
		参数:
			data_dir (str): 数据目录
			tile_pattern (str): 文件匹配模式
			channel_indices (list): 要使用的通道索引列表
			channel_names (list, optional): 通道名称列表，便于可视化和调试
			transform (callable, optional): 数据转换函数
			normalize (bool): 是否标准化数据
			means_stds_file (str, optional): 存储均值和标准差的文件路径
			grid_size (int): 切片大小
			log_transform (bool): 是否启用对数变换
			log_transform_indices (list): 需要对数变换的通道索引列表
		"""
		self.data_dir = data_dir
		self.tiles = glob.glob(os.path.join(data_dir, tile_pattern))
		self.subset = subset
		self.subset_cols = subset_cols if subset_cols else [f"channel_{i}" for i in subset]
		self.transform = transform
		self.normalize = normalize
		self.log_transform = log_transform
		# self.log_transform_indices = log_transform_indices if log_transform_indices else []
		self.grid_size = grid_size


		self.means = means
		self.stds = stds
		self.mins = mins

		
	

	def __len__(self):
		return len(self.tiles)

	def _standardize_input(self, values):
		"""按通道标准化输入数据"""
		standardized = values.copy()
		
		for i, col in enumerate(self.subset_cols):
			values[i, :, :] = self.standardize_channel(values[i, :, :], self.means[i], self.stds[i], col, self.mins[i], self.log_transform)
		return values

	def standardize_channel(self, values, mean, std, col, mins, log_transform):
			if col in ['C03', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']:
				return (values - mean) / std
			elif col in ['C01', 'C02', 'C04']:
				if log_transform:
					return (np.where(values == 0., np.log(mins - 1e-10), np.log(values)) - mean) / std
				else:
					return (values - mean) / std
			else:
				print('Unknown channel {}'.format(col))
				return None
			

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		# 获取文件名
		filename = self.tiles[idx]
		id = os.path.basename(filename)
		
		# 读取数据
		with xr.open_dataset(filename) as ds:
			# 提取所需通道数据
			if self.subset is not None:
				grid_values = np.array(ds[self.subset_cols].to_array().values)
			else:
				grid_values = np.array(ds.to_array().values)

			cld_mask = np.array(ds['CLM'].values) 
        
			center = np.array(ds['center'].values)  # 格式可能为 [lat, lon]

			if self.normalize:
				grid_values = self._standardize_input(grid_values)


		
		# 创建样本字典
		sample = {
			'data': grid_values.astype(np.float32),
			'cld_mask': cld_mask.astype(int),
			'center': center.astype(np.float64),
			'id': id,
		}
		
		# 应用转换
		if self.transform:
			sample['data'] = self.transform(sample['data'])
		
		return sample

# endregion



