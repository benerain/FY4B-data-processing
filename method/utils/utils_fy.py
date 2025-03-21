import xarray as xr
import os
import glob
import pandas as pd

import numpy as np
np.int = int

import itertools
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap 

from method.utils.pytorch_class_fy import FY4BTilesDataset
# # 设置matplotlib的全局字体配置
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'sans-serif'

# from method.utils.FYutils import FYGlobalTilesDataset

# import torch
# import torch.nn as nn
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


# from method.models.models import ConvAutoEncoder
# from joblib import load




MIN_N_TILES = 3000
MAX_WIDTH, MAX_HEIGHT = 1725,1002

def get_tile_offsets(tile_size=128):
	"""
	Returns index of center depending on the tile size.
	In the case of an even tile size, the center is defined by (tile_size//2, tile_size//2-1).

	:param tile_size: Square size of the desired tile.
	:returns: Offset sizes.
	"""
	offset = tile_size // 2
	offset_2 = offset
	if not tile_size % 2:
		offset_2 -= 1
	return offset, offset_2



def get_sampling_mask(mask_shape=(MAX_WIDTH, MAX_HEIGHT), tile_size=256):
	"""
	Returns a mask of allowed centers for the tiles to be sampled.
	The center of an even size tile is considered to be the point
	at the position (size // 2, size // 2 - 1) within the tile.

	:param mask_shape: Size of the input file ie mask size.
	:param tile_size: Size of the desired tiles.
	:returns: mask
	"""
	mask = np.ones(mask_shape, dtype=np.uint8)
	offset, offset_2 = get_tile_offsets(tile_size)
	# must not sample tile centers in the borders, so that tiles keep to required shape
	mask[:, :offset] = 0
	mask[:, -offset_2:] = 0
	mask[:offset, :] = 0
	mask[-offset_2:, :] = 0
	return mask



def fill_in_values(data, fill_in=True):
    """
    Change fill-in values in grids for FY4B channels:
    C01-C06: 反射率通道 (0值填充)
    C07-C15: 亮温通道 (100K填充)
    CLM:     云掩膜通道 (127填充)
    
    通道顺序需要严格对齐数据数组的维度顺序:
    [C01, C02, C03, C04, C05, C06, C07, C08, C09, 
     C10, C11, C12, C13, C14, C15, CLM] (共16个通道)

    :param data: 三维数据数组 (channels, y, x)
    :param fill_in: 是否用预定义值填充NaN
    :returns: 填充后的数据和填充值数组
    """
    if fill_in:
        # 定义每个通道的填充值 (按顺序对齐)
        filling_in = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # C01-C06: 反射率通道
            100.0, 100.0, 100.0, 100.0,     # C07-C10
            100.0, 100.0, 100.0, 100.0,     # C11-C14
            100.0               
        ], dtype=np.float64)
        
        # 确保通道数量一致
        assert data.shape[0] == 15, "需要15个通道(C01-C15)"
        
        # 逐通道填充
        for i in range(15):
            data[i, :, :] = np.where(
                np.isnan(data[i, :, :]),
                filling_in[i],
                data[i, :, :]
            )
    else:
        filling_in = np.full((data.shape[0],), np.nan, dtype=np.float64)
    
    return data, filling_in




def read_channel(file, channel):
	"""
	Read in from the loaded file the corresponding channel.

	:param file: xarray.Dataset object.
	:param channel: Name of the channel.
	:returns: Data array of the channel.
	"""
	# Get data
	return file[channel].data


def get_channels_cloud_mask(filename): 
    """
    从二级文件中返回以下内容：
    - 通道数据：大小为 (n_channels, HEIGHT, WIDTH) 的 numpy 数组，包含所有二级通道
    - 云掩膜：大小为 (HEIGHT, WIDTH) 的 numpy 数组，表示云掩膜

    :param filename: 文件名字符串。
    :returns: 通道数据、云掩膜、纬度、经度、填充值、标签掩膜及云层底部高度（如有）。
    """
    level_data = None
    try:
		# Open file and get rid of time dimension   
        level_data = xr.open_dataset(filename).isel(time=0)
    except IndexError as err:
        print('    Could not open file: {}'.format(err))


    if level_data is not None:
        # 纬度和经度
        latitude = read_channel(level_data, 'latitude')
        longitude = read_channel(level_data, 'longitude')

		# Retrieve corresponding channels data
        c01 = read_channel(level_data, 'C01')
        c02 = read_channel(level_data, 'C02')
        c03 = read_channel(level_data, 'C03')
        c04 = read_channel(level_data, 'C04')
        c05 = read_channel(level_data, 'C05')
        c06 = read_channel(level_data, 'C06')
        c07 = read_channel(level_data, 'C07')  
        c08 = read_channel(level_data, 'C08')
        c09 = read_channel(level_data, 'C09')
        c10 = read_channel(level_data, 'C10')
        c11 = read_channel(level_data, 'C11')
        c12 = read_channel(level_data, 'C12')
        c13 = read_channel(level_data, 'C13')
        c14 = read_channel(level_data, 'C14')
        c15 = read_channel(level_data, 'C15')

        
        channels = np.stack([ c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15])
        channels, filling_in = fill_in_values(channels, fill_in=True)

        raw_clm = read_channel(level_data, 'CLM')
        # print(raw_clm)
        cloud_mask = xr.where(raw_clm == 0, 1, 0).astype(np.uint8)
        

        return channels.astype(np.float32), cloud_mask.astype(np.uint8), \
            latitude.astype(np.float32), longitude.astype(np.float32),\
            filling_in.astype(np.float16)
    
    else:
        return None, None, None, None, None, None
    
def setup_xarray_tile(tile, cbh=None, center=None, tile_size=None):
     """
     将tile数据转换为xarray数据集
     
     :param tile: tile数据，形状为[channels, height, width]
     :param cbh: 云底高度数据，如果有的话
     :param center: 中心坐标
     :param tile_size: 瓦片大小，如果为None则从数据推断
     :returns: xarray数据集
     """
     # 如果未指定tile_size，从数据推断
     if tile_size is None:
          tile_size = tile.shape[1]  # 假设tile是正方形的
     
     # 初始化数据集字典
     dataset_dict = {
          'C01': (['lat', 'lon'], tile[0, :, :]),
          'C02': (['lat', 'lon'], tile[1, :, :]),
          'C03': (['lat', 'lon'], tile[2, :, :]),
          'C04': (['lat', 'lon'], tile[3, :, :]),
          'C05': (['lat', 'lon'], tile[4, :, :]),
          'C06': (['lat', 'lon'], tile[5, :, :]),
          'C07': (['lat', 'lon'], tile[6, :, :]),
          'C08': (['lat', 'lon'], tile[7, :, :]),
          'C09': (['lat', 'lon'], tile[8, :, :]),
          'C10': (['lat', 'lon'], tile[9, :, :]),
          'C11': (['lat', 'lon'], tile[10, :, :]),
          'C12': (['lat', 'lon'], tile[11, :, :]),
          'C13': (['lat', 'lon'], tile[12, :, :]),
          'C14': (['lat', 'lon'], tile[13, :, :]),
          'C15': (['lat', 'lon'], tile[14, :, :]),
          'CLM': (['lat', 'lon'], tile[15, :, :]),
     }
     
     # 如果cbh不为None且是一个有效的列表，添加cbh相关字段
     if cbh is not None and isinstance(cbh, list) and len(cbh) >= 2:
          dataset_dict['cbh'] = (['lat', 'lon'], cbh[0])
          dataset_dict['cbh_center'] = cbh[1]
     
     # 添加center字段（如果提供）
     if center is not None:
          dataset_dict['center'] = center
     
     # 创建并返回xarray数据集
     tile_xr = xr.Dataset(
          dataset_dict,
          coords={
               'lat': np.arange(tile_size),
               'lon': np.arange(tile_size)
          }
     )
     return tile_xr



def save_tiles_nc(swath_name, tiles, ddir, center, cbh=None):
	"""
	Save tiles to destination directory as xarray.Dataset/netCDF objects.

	:param swath_name: Swath id name.
	:param tiles: List containing the extracted tiles.
	:param ddir: Destination directory.
	:returns: None
	"""
	if not os.path.exists(ddir):
		os.makedirs(ddir)

	# Remove existing tiles
	list_tiles = glob.glob(ddir + '*.nc')
	for t in list_tiles:
		os.remove(t)

	for i, tile in enumerate(tiles, 1):
		cbh_data = None
		if cbh is not None and isinstance(cbh, list) and len(cbh) == 2 and len(cbh[0]) >= i and len(cbh[1]) >= i:
			cbh_data = [cbh[0][i - 1], cbh[1][i - 1]]
		setup_xarray_tile(
			tile=tile, 
			cbh=cbh_data,
			center=center[i - 1]).to_netcdf(ddir + "{}_{}.nc".format(swath_name, i))



def extract_cloudy_tiles_swath(
		swath_array, cloud_mask, latitude, longitude,
		fill_values, cbh, regular_sampling=False, sampling_step='wide',
		n_tiles=20, tile_size=128, cf_threshold=0.3, verbose=False):
	"""
	The script will use a cloud_mask channel to mask away all non-cloudy data.
	The script will then select all tiles from the cloudy areas where the cloud fraction is at least cf_threshold.

	:param swath_array: input numpy array from MODIS of size (nb_channels, w, h)
	:param cloud_mask: 2d array of size (w, h) marking the cloudy pixels
	:param latitude: Latitude array.
	:param longitude: Longitude array.
	:param fill_values: Values to fill in the missing values of the cloud properties.
	:param cbh: input array of the cbh retrieval (dimension (w, h)) if available.
	:param regular_sampling: Tiles to be sampled regularly (according to some step size) or randomly.
	:param sampling_step: Which step size to use when sampling regularly the tiles. 'wide' = 128 km, 'regular' = 64 km
	and 'fine' = 10 km. Values can be adapted if necessary.
	:param n_tiles: Number of tiles to sample from the swath.
	:param tile_size: size of the tile selected from within the image
	:param cf_threshold: cloud fraction threshold to apply when filtering cloud scenes.
	:param verbose: Display information or not.
	:return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles and corresponding cbh label (nb_tiles,)
	"""
	# Compute distances from tile center of tile upper left and lower right corners
	offset, offset_2 = get_tile_offsets(tile_size)

	if regular_sampling:

		tile_centers = []
		# Sampling centers
		step_size = tile_size // 2 if sampling_step == 'wide' else (64 if sampling_step == 'regular' else 10)
		idx_w = np.arange(start=2 * tile_size, stop=cloud_mask.shape[0] - 2 * tile_size, step=step_size)
		idx_h = np.arange(start=2 * tile_size, stop=cloud_mask.shape[1] - 2 * tile_size, step=step_size)
		for c_w, c_h in itertools.product(idx_w, idx_h):
			tile_centers.append([c_w, c_h])
			# tile_centers.append([latitude[c_w, c_h], longitude[c_w, c_h]])
		tile_centers = np.array(tile_centers)

	else:

		# Mask out borders not to sample outside the swath
		allowed_pixels = get_sampling_mask(swath_array.shape[1:], tile_size)

		# Tile centers will be sampled from the cloudy pixels that are not in the borders of the swath
		cloudy_label_pixels = np.logical_and.reduce([allowed_pixels.astype(bool), cloud_mask.astype(bool)])
		cloudy_label_pixels_idx = np.where(cloudy_label_pixels == 1)
		cloudy_label_pixels_idx = list(zip(*cloudy_label_pixels_idx))

		# Number of tiles to sample from
		number_of_tiles = min(MIN_N_TILES, len(cloudy_label_pixels_idx))
		# Sample without replacement
		tile_centers_idx = np.random.choice(np.arange(len(cloudy_label_pixels_idx)), number_of_tiles, False)
		cloudy_pixels_idx = np.array(cloudy_label_pixels_idx)
		tile_centers = cloudy_pixels_idx[tile_centers_idx]

	positions, centers, centers_lat_lon, tiles, cbh_values, cbh_values_center = [], [], [], [], [], []

	for center in tile_centers:

		center_w, center_h = center

		w1 = center_w - offset
		w2 = center_w + offset_2 + 1
		h1 = center_h - offset
		h2 = center_h + offset_2 + 1

		# Check cloud fraction in tile
		cf = cloud_mask[w1:w2, h1:h2].sum() / (tile_size * tile_size)

		# Check missing values in tile
		mv = (swath_array[:, w1:w2, h1:h2] == fill_values[:, np.newaxis, np.newaxis]).sum(axis=(1, 2)) / (
				tile_size * tile_size)

		# If cloud fraction in the tile is higher than cf_threshold then store it
		if (cf >= cf_threshold) and all(mv < 1.):
			tile = swath_array[:, w1:w2, h1:h2]
			tile_position = ((w1, w2), (h1, h2))
			# tile_cbh_value_center = cbh[center_w, center_h]
			# tile_cbh_value = cbh[w1:w2, h1:h2]

			# Stack with cloud mask
			tile = np.concatenate([tile, cloud_mask[np.newaxis, w1:w2, h1:h2]], axis=0)

			positions.append(tile_position)
			centers.append(center)
			centers_lat_lon.append((latitude[center_w, center_h], longitude[center_w, center_h]))
			tiles.append(tile)
			# cbh_values.append(tile_cbh_value)
			# cbh_values_center.append(tile_cbh_value_center)

	if len(tiles) > 0:
		n_tiles = len(tiles) if regular_sampling else min(n_tiles, len(tiles))
		print('    {} extracted tiles'.format(n_tiles)) if verbose else None
		tiles = np.stack(tiles[:n_tiles])
		positions = np.stack(positions[:n_tiles])
		centers = np.stack(centers[:n_tiles])
		centers_lat_lon = np.stack(centers_lat_lon[:n_tiles])
		# cbh_values = np.stack(cbh_values[:n_tiles])
		# cbh_values_center = np.stack(cbh_values_center[:n_tiles])

		return tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center

	else:
		print('    No valid tiles could be extracted from the swath.') if verbose else None
		return None, None, None, None, None, None


def sample_tiles_swath(filename, dest_dir, regular_sampling=False, sampling_step='wide', n_tiles=20, tile_size=256, cf_threshold=0.3, verbose=True):
	"""
	Create tiles from the swath.
	Save tiles in corresponding folder.

	:param filename: Filename of the swath file.
	:param dest_dir: Destination directory to save the extracted tiles.
	:param regular_sampling: Tiles to be sampled regularly (according to some step size) or randomly.
	:param sampling_step: Which step size to use when sampling regularly the tiles. 'wide' = 128 km, 'regular' = 64 km
	and 'fine' = 10 km. Values can be adapted if necessary.
	:param n_tiles: Number of tiles to sample from the swath.
	:param tile_size: size of the tile selected from within the image
	:param cf_threshold: cloud fraction threshold to apply when filtering cloud scenes.
	:param verbose: Display information or not.
	:returns: Centers of extracted tiles as array (latitude, longitude).
	"""

	# Swath name
	swath_name = filename.split('/')[-1][:-3]
	print('Swath file {}'.format(swath_name)) if verbose else None

	# Extract channels data and cloud_mask from MODIS file
	print('    Extracting channels and cloud mask data...') if verbose else None
	swath_array, cloud_mask, latitude, longitude, fill_values = get_channels_cloud_mask(
		filename=filename)
	
	# In case the file could not be opened
	if swath_array is None:
		print('    Error - File {} could not be opened...'.format(swath_name)) if verbose else None
		return None

	# Extract cloudy tiles from the swath data
	print('    Extracting tiles for swath {} ...'.format(swath_name)) if verbose else None
	tiles, positions, centers, centers_lat_lon, cbh_values, cbh_values_center = extract_cloudy_tiles_swath(
		swath_array=swath_array,
		cloud_mask=cloud_mask,
		latitude=latitude,
		longitude=longitude,
		fill_values=fill_values,
		cbh=None,
		regular_sampling=regular_sampling,
		sampling_step=sampling_step,
		n_tiles=n_tiles,
		tile_size=tile_size,
		cf_threshold=cf_threshold,
		verbose=verbose)

	# Save tiles
	if tiles is not None:
		print('    Saving tiles to output directory ...') if verbose else None
		
		# 检查cbh_values和cbh_values_center是否有效
		cbh_param = None
		if (cbh_values is not None and cbh_values_center is not None and 
			isinstance(cbh_values, np.ndarray) and isinstance(cbh_values_center, np.ndarray) and
			len(cbh_values) > 0 and len(cbh_values_center) > 0):
			cbh_param = [cbh_values, cbh_values_center]
		
		save_tiles_nc(
			swath_name=swath_name, 
			tiles=tiles, 
			center=centers_lat_lon,
			cbh=cbh_param, 
			ddir=dest_dir)

	return centers_lat_lon

def plot_tile(tile):
    """
    Plot all 15 FY-4B satellite channels plus cloud mask in a 3x6 grid layout.
    
    :param tile: 文件路径或xarray Dataset对象
    :returns: None
    """
    # 设置matplotlib的中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果传入的是字符串（文件路径），则打开文件
    if isinstance(tile, str):
        try:
            tile = xr.open_dataset(tile)
            print(f"成功打开文件: {tile}")
        except Exception as e:
            print(f"打开文件时出错: {e}")
            return None
    
    # 打印数据集中的所有变量，帮助调试
    print(f"数据集中的变量: {list(tile.data_vars)}")
    
    # 定义通道列表和它们的类型分组（注意使用小写变量名）
    channels = [
        # 可见光和近红外通道（反射率）
        {"var": "C01", "name": "0.47μm (蓝)", "type": "可见光", "cmap": "Blues_r"},
        {"var": "C02", "name": "0.65μm (红)", "type": "可见光", "cmap": "Blues_r"},
        {"var": "C03", "name": "0.83μm (近红外)", "type": "近红外", "cmap": "Blues_r"},
        {"var": "C04", "name": "1.37μm (水汽检测)", "type": "近红外", "cmap": "Blues_r"},
        {"var": "C05", "name": "1.61μm (雪/冰检测)", "type": "近红外", "cmap": "Blues_r"},
        {"var": "C06", "name": "2.22μm (云粒子)", "type": "近红外", "cmap": "Blues_r"},
        
        # 水汽通道（亮温）
        {"var": "C09", "name": "6.25μm (高层水汽)", "type": "水汽", "cmap": "Blues_r"},
        {"var": "C10", "name": "6.95μm (中层水汽)", "type": "水汽", "cmap": "Blues_r"},
        
        # 红外窗区通道（亮温）
        {"var": "C07", "name": "3.75μm (高)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C08", "name": "3.75μm (低)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C11", "name": "7.42μm (低对流层)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C12", "name": "8.55μm (总水量)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C13", "name": "10.8μm (清晰窗区)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C14", "name": "12.0μm (脏窗区)", "type": "红外", "cmap": "Blues_r"},
        {"var": "C15", "name": "13.3μm (CO2吸收)", "type": "红外", "cmap": "Blues_r"},
    ]
    
    # 创建3x6网格的子图（每行3个，共6行，最多可放18个子图）
    # 增加图形大小
    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(18, 36))
    axs = axs.flatten()  # 将2D数组展平为1D，便于索引
    
    # 设置图表的整体标题
    fig.suptitle('FY-4B 卫星通道数据和云掩膜', fontsize=26, y=0.995)
    
    # 大幅减少子图之间的间距
    plt.subplots_adjust(hspace=0.12, wspace=0.05)
    
    # 绘制15个通道数据
    for i, channel_info in enumerate(channels):
        # 获取变量
        channel_var = channel_info["var"]
        
        # 检查该变量是否存在于数据集中
        if channel_var in tile:
            channel_data = tile[channel_var]
            # 检查是否有时间维度，如果有则选择第一个时间点
            if 'time' in channel_data.dims and len(channel_data.time) > 0:
                channel_data = channel_data.isel(time=0)
            
            # 动态计算vmin和vmax以确保数据可见
            valid_data = channel_data.values[~np.isnan(channel_data.values)]
            if len(valid_data) > 0:
                # 使用百分位数计算范围，避免极值影响
                vmin = np.nanpercentile(valid_data, 1)
                vmax = np.nanpercentile(valid_data, 99)
            else:
                # 如果没有有效数据，使用默认值
                vmin, vmax = 0, 1
                
            # 使用配置的颜色映射和动态范围
            im = axs[i].imshow(channel_data, 
                              aspect='equal', 
                              cmap=channel_info["cmap"], 
                              vmin=vmin, 
                              vmax=vmax)
            
            # 添加单位信息，简化标签
            units = "%" if "可见光" in channel_info["type"] or "近红外" in channel_info["type"] else "K"
            axs[i].set_title(f'{channel_info["name"]}\n({channel_info["type"]})', fontsize=12, pad=3)
            axs[i].axis('off')
        else:
            axs[i].text(0.5, 0.5, f'通道 {channel_var} 不可用', 
                       horizontalalignment='center', verticalalignment='center')
            axs[i].axis('off')
    
    # 在第16个位置绘制云掩膜
    cloud_mask_var = 'CLM'
    if cloud_mask_var in tile:
        mask_data = tile[cloud_mask_var]
        if 'time' in mask_data.dims and len(mask_data.time) > 0:
            mask_data = mask_data.isel(time=0)
        
        # 将mask_data转换为0和1的二值图像
        # 0表示晴空（深蓝色 ('#191966')），1表示云（灰色 ('#646464')）.
        colors = [ '#191966','#646464']


        cmap_custom = ListedColormap(colors)

        # 显示云掩膜图像
        im = axs[15].imshow(mask_data, aspect='equal', cmap=cmap_custom, vmin=0, vmax=1)
        
        # 在图像上添加图例，而不是使用colorbar
        # 创建小色块代表颜色
        cloud_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#191966')
        no_cloud_patch = plt.Rectangle((0, 0), 1, 1, facecolor='#646464')
        
        # 添加图例，放置于右上角
        axs[15].legend([cloud_patch, no_cloud_patch], ['晴空', '云'], 
                      loc='upper right', bbox_to_anchor=(1.0, 1.0),
                      fontsize=12)
        
        # 设置标题
        axs[15].set_title('云掩膜', fontsize=12, color='blue', pad=3)
        axs[15].axis('off')
    else:
        axs[15].text(0.5, 0.5, 'CLM 云掩膜不可用', 
                    horizontalalignment='center', verticalalignment='center')
        axs[15].axis('off')
    
    # 隐藏未使用的子图
    for i in range(16, len(axs)):
        axs[i].axis('off')
        axs[i].set_visible(False)
    
    # 调整布局，减少填充
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()
    
    # 关闭数据集，释放资源
    if isinstance(tile, xr.Dataset):
        tile.close()

# endregion

# region 计算通道统计量


# TODO: 计算全部文件的通道统计量，当前只有一个文件
def calculate_channel_stats(data_dir: str, output_file: str, log_channels=None) -> None:
    """
    遍历 data_dir 目录下所有 .nc 文件，计算每个文件中 FY4B 卫星通道 (C01~C15) 的统计量：
      - 均值 (mean)
      - 标准差 (std)
      - 最小值 (min)
      - 最大值 (max)
      - 对数均值 (log_mean) 与对数标准差 (log_std)：仅对 log_channels 指定的通道（如 ["C01", "C02", "C04"]）进行，
        仅对数据大于 0 的像元计算；其他通道输出 nan。

  
 
    """
    if log_channels is None:
        log_channels = []
    log_channels_set = set(log_channels)
    
    # 获取 data_dir 下所有 .nc 文件
    file_paths = glob.glob(os.path.join(data_dir, "*.nc"))
    if not file_paths:
        print(f"在目录 {data_dir} 下未找到 .nc 文件。")
        return

    # 如果 output_file 不是绝对路径，则构造完整路径
    if not os.path.isabs(output_file):
        output_path = os.path.join(data_dir, output_file)
    else:
        output_path = output_file

    # 打开输出文件（覆盖方式）
    with open(output_path, "w", encoding="utf-8") as out_f:
        # 遍历所有文件
        for file_path in file_paths:
            try:
                with xr.open_dataset(file_path) as ds:
                    # 定义通道顺序：C01～C15
                    channels_order = [f"C{i:02d}" for i in range(1, 16)]
                    
                    # 初始化统计列表
                    means, stds, mins_, maxs_ = [], [], [], []
                    log_means, log_stds = [], []
                    
                    for chan in channels_order:
                        if chan not in ds:
                            # 若当前通道不存在，则输出 nan
                            means.append(np.nan)
                            stds.append(np.nan)
                            mins_.append(np.nan)
                            maxs_.append(np.nan)
                            log_means.append(np.nan)
                            log_stds.append(np.nan)
                            continue

                        data = ds[chan]
                        means.append(data.mean().item())
                        stds.append(data.std().item())
                        mins_.append(data.min().item())
                        maxs_.append(data.max().item())

                        if chan in log_channels_set:
                            # 仅对大于 0 的值取自然对数进行统计
                            data_pos = data.where(data > 0)
                            count_pos = data_pos.count().item()
                            if count_pos > 0:
                                lm = np.log(data_pos).mean().item()
                                ls = np.log(data_pos).std().item()
                            else:
                                lm, ls = np.nan, np.nan
                            log_means.append(lm)
                            log_stds.append(ls)
                        else:
                            log_means.append(np.nan)
                            log_stds.append(np.nan)
                    
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    # 构造输出内容
                    lines = []
                    lines.append(f"File: {os.path.basename(file_path)}")
                    lines.append(f"RUN time {now_str}")
                    lines.append("Statistics for FY4B satellite channels")
                    
                    # 输出通道标签：channel_0 ~ channel_14 对应 C01 ~ C15
                    channel_labels = [f"C{i:02d}" for i in range(1, 16)]
                    lines.append(" ".join(channel_labels))
                    
                    lines.append(" ".join(f"{v:.6f}" for v in means))
                    lines.append(" ".join(f"{v:.6f}" for v in stds))
                    lines.append(" ".join(f"{v:.6f}" for v in mins_))
                    lines.append(" ".join(f"{v:.6f}" for v in maxs_))
                    lines.append(" ".join(f"{v:.6f}" for v in log_means))
                    lines.append(" ".join(f"{v:.6f}" for v in log_stds))
                    
                    out_str = "\n".join(lines) + "\n\n"
                    
                    # 打印并写入文件
                    print(out_str)
                    out_f.write(out_str)
                    
                    print(f"Processed file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    print(f"Stats file saved to: {output_path}")

def load_means_stds(file_path, subset_channels =None, log_transform_channels=None):
    """
    加载全局 FY4B 统计文件，支持通道子集和对数变换
    
    Args:
        file_path (str): 统计文件路径（全局）
        subset_channels (list): 需加载的通道索引（如 [0,1,2] 对应 C01-C03），默认为None表示选择所有通道
        log_transform_channels (list): 需应用对数变换的通道索引（如 [2]）

    Returns:
        means, stds, mins, maxs (np.array): 处理后的统计量数组
    """
    # 读取文件内容
    df_means_stds = pd.read_csv( 
		file_path, header=0, skiprows=3, sep=' ')
    
    Means = df_means_stds.iloc[0].values
    Stds = df_means_stds.iloc[1].values
    Mins = df_means_stds.iloc[2].values
    Maxs = df_means_stds.iloc[3].values
    Means_log = df_means_stds.iloc[4].values
    Stds_log = df_means_stds.iloc[5].values
    if log_transform_channels is not None:
        for idx in log_transform_channels:
            Means[idx] = Means_log[idx]
            Stds[idx] = Stds_log[idx]
            
    # 如果subset_channels为None，选择所有通道
    if subset_channels is None:
        subset_channels = list(range(len(Means)))
        
    means = Means[subset_channels]
    stds = Stds[subset_channels]
    mins = Mins[subset_channels]
    maxs = Maxs[subset_channels]
    return means, stds, mins, maxs
    


def load_data_tiles(ddir, tile_names, means_stds_file):
	"""
	Load pytorch.Dataset object for the tiles.

	:param ddir: Data directory of the tile files.
	:param tile_names: Regex for tiles to use from the data directory.
	:param means_stds_file: File containing means and standard deviations of the different cloud properties.
	:returns: Dataset object from ModisGlobalTilesDataset and corresponding pytorch.DataLoader.
	"""
	# Cloud properties to use in input
	param_cols = ['cloud_top_height', 'cloud_optical_thickness', 'cloud_water_path']
	subset = [1, 3, 4]
	# Additional parameters for the pytorch.Dataset class ModisGlobalTilesDataset
	ext = 'nc'
	subscale = False
	grid_size = 256
	log_transform = True
	log_transform_indices = [0,1,3]
	normalize = True
	means, stds, mins, _ = load_means_stds(file=means_stds_file, log_transform=log_transform)
	dataset = FY4BTilesDataset(
		data_dir=ddir, tile_pattern=ext, channel_indices=subset, channel_names=param_cols,
		transform=None,  grid_size=grid_size,
		normalize=normalize, mean=means, std=stds, min=mins, log_transform=log_transform,
        log_transform_indices=log_transform_indices)
	batch_size = 64 if len(dataset) >= 64 else len(dataset)
	dataloader = DataLoader(dataset, batch_size=batch_size,
							shuffle=False, drop_last=False, num_workers=1)
	return dataset, dataloader

    

# endregion
