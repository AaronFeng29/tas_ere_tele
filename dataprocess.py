# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:56:30 2025

@author: Aaron
"""

import xarray as xr
import numpy as np
import os
from datetime import datetime

# 1. 设置输入路径
input_dir = "../../ERA5-tas1000"  # 替换为实际路径

# 2. 遍历文件并读取温度数据
temp_list = []
dates = []

for filename in sorted(os.listdir(input_dir)):  # 确保按日期顺序读取
    if filename.endswith(".nc"):
        # 解析日期（假设文件名格式为YYYYMMDD.nc）
        print(filename)
        date_str = filename.split(".")[0]
        date = datetime.strptime(date_str, "%Y%m%d")
        dates.append(date)
        
        # 读取温度数据（假设变量名为"temperature"）
        file_path = os.path.join(input_dir, filename)
        with xr.open_dataset(file_path) as ds:
            temp = ds["t"].values  # 提取温度值（纬度×经度）
            temp_list.append(temp)

# 3. 转换为NumPy数组（时间×纬度×经度）
temperature_array = np.stack(temp_list, axis=0)
# np.save("origin_data.npy",temperature_array)
# 4. 验证结果
print(f"合并后的数组形状：{temperature_array.shape}")
print(f"数据类型：{temperature_array.dtype}")
print(f"时间范围：{dates[0].strftime('%Y-%m-%d')} 至 {dates[-1].strftime('%Y-%m-%d')}")