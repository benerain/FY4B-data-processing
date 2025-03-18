import os
import glob
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



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


# def standardize_channel(array, mean, std, channel, epsilon, log_transform):
# 	"""
# 	Standardize input array along channels with mean/std.

# 	:param array: Data array.
# 	:param mean: Mean of channel.
# 	:param std: Standard deviation of channel.
# 	:param channel: Channel name.
# 	:param epsilon: Minimum value of the corresponding channel.
# 	:param log_transform: If channel is to be normalized with the log.
# 	:returns: Standardized data array.
# 	"""
# 	if channel in ['cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature']:
# 		return (array - mean) / std
# 	elif channel in ['cloud_optical_thickness', 'cloud_water_path']:
# 		if log_transform:
# 			return (np.where(array == 0., np.log(epsilon - 1e-10), np.log(array)) - mean) / std
# 		else:
# 			return (array - mean) / std
# 	else:
# 		print('Unknown channel {}'.format(channel))
# 		return None

def standardize_channel(array, mean, std, channel, epsilon, log_transform):
    """
    Standardize input array along channels with mean/std.

    :param array: Data array.
    :param mean: Mean of channel.
    :param std: Standard deviation of channel.
    :param channel: Channel name (例如 'CER', 'CTH', 'CTT', 'COT', 'LWP').
    :param epsilon: Minimum value of the corresponding channel.
    :param log_transform: If channel is to be normalized with the log.
    :returns: Standardized data array.
    """
    # 对压力、高度、温度等通道（例如 CER, CTH, CTT）使用常规归一化
    if channel in ['CER', 'CTH', 'CTT']:
        return (array - mean) / std
    # 对光学厚度和云水路径（例如 COT, LWP），可选用对数归一化
    elif channel in ['COT', 'LWP']:
        if log_transform:
            # 避免取对数时出现 log(0)，将0或负值处理为 epsilon（减去一个微小值以确保数值稳定）
            return (np.where(array == 0., np.log(epsilon - 1e-10), np.log(array)) - mean) / std
        else:
            return (array - mean) / std
    else:
        print('Unknown channel {}'.format(channel))
        return None


class FYGlobalTilesDataset(Dataset):
	"""
	Pytorch 数据集类，用于加载 FY 平台生成的云属性瓦片。
  
	"""

	def __init__(self, ddir, ext, tile=None, subset=None, subset_cols=None,
	             transform=None, log_transform=False, normalize=False, mean=None, std=None,min=None,
	             subscale=False, grid_size=None, get_ground_cbh=False):
		"""

        :param ddir: 存放瓦片文件的目录。
        :param ext: 文件扩展名（例如 "nc"）。
        :param tile: 文件名前缀（默认为 "tile"）。
        :param subset: 是否仅使用部分通道（例如 0,1,2,...）。
        :param subset_cols: 使用的通道名称列表，
        :param transform: 应用于样本数据的 pytorch 转换。
        :param log_transform: 是否对部分通道（如 COT、CWP）进行对数变换。
        :param normalize: 是否对通道数据进行归一化。
        :param mean: 各通道均值（归一化时用）。
        :param std: 各通道标准差（归一化时用）。
        :param min: 各通道最小值（归一化时用）。
        :param subscale: 是否对子瓦片进行尺寸缩放。
        :param grid_size: 瓦片尺寸（如 128）。
        :param get_ground_cbh: 是否加载额外的云底高度信息。
		"""
		# Directory with tiles files
		self.ddir = ddir
		# Extension to use for the files
		self.ext = ext
		# Filename to use
		self.tile = 'tile' if tile is None else tile
		# Column subset
		self.subset = subset
		self.subset_cols = subset_cols
		# Transforms to apply
		self.transform = transform
		# Define parameters to transform of input data
		self.subscale = subscale
		self.grid_size = grid_size

		self.normalize = normalize
		self.mean = mean
		self.std = std
		self.min = min
		self.log_transform = log_transform
		self.get_ground_cbh = get_ground_cbh
		self.file_paths = glob.glob(os.path.join(self.ddir, self.tile + '*.' + self.ext))
		self.file_paths.sort()

		if len(self.file_paths) == 0:
			raise FileNotFoundError("No {} files found in directory {} ...".format(self.ext, self.ddir))

	def __len__(self):
		return len(self.file_paths)

	def _standardize_input(self, values):
		for i, col in enumerate(self.subset_cols):
			values[i, :, :] = standardize_channel(values[i, :, :], self.mean[i], self.std[i], col, self.min[i], self.log_transform)
		return values

	def __getitem__(self, idx):

		# 如果传入的索引 idx 是一个 PyTorch 张量（tensor），则将其转换为 Python 列表，以便后续用作列表索引。
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# 获取文件名和 ID
		filename = self.file_paths[idx]
		id = os.path.basename(filename) 

		data = read_nc(filename=filename)

		# 设置了只提取部分通道
		if self.subset:
			grid_values = np.array(data[self.subset_cols].to_array().values)
		else:
			grid_values = np.array(data.to_array().values)
		
		# 从数据中提取名为 'CLM' 的变量（云掩膜数据），并转换成 NumPy 数组。
		cld_mask = np.array(data['CLM'].values)
		# Get center
		center = np.array(data['center'].values)

		# Get cloud base height from calipso/cloudsat retrievals
		if self.get_ground_cbh:
			ground_cbh = np.array(data['cloud_base_height'].values)
			ground_cbh_center = np.array(data['cloud_base_height_center'].values)

		# 如果需要对子瓦片进行尺寸缩放
		if self.subscale:
			grid_values = subscalegrid(self.grid_size, grid_values)
			cld_mask = subscalegrid(self.grid_size, cld_mask)
		# 归一化处理
		if self.normalize:
			grid_values = self._standardize_input(grid_values)

		sample = {'data': grid_values.astype(np.float64),
		          'cld_mask': cld_mask.astype(int),
		          'center': center.astype(np.float64),
		          'id': id,
		          'ground_cbh': ground_cbh.astype(np.float64) if self.get_ground_cbh else -1,
		          'ground_cbh_center': ground_cbh_center.astype(np.float64) if self.get_ground_cbh else -1}

		if self.transform:
			sample['data'] = self.transform(sample['data'])

		return sample

# endregion
