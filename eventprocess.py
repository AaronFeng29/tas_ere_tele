# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:56:30 2025

@author: Aaron
"""

import numpy as np
import scipy.stats as st

# 加载数据并预处理
globe_diff = np.load("./deseason_data.npy").squeeze().reshape([15695, 73, 144])
globe_diff = np.diff(globe_diff, axis=0)  # 计算日温差
print("Data shape:", globe_diff.shape)

# 展平为时间序列x格点矩阵
time_steps, lat, lon = globe_diff.shape
globe_flat = globe_diff.reshape(time_steps, -1)  # (15694, 10512)

# 定义处理函数
def process_quantile(data, perc):
    # 计算分位数阈值
    thresholds = np.apply_along_axis(lambda x: st.scoreatpercentile(x, perc), axis=0, arr=data)
    
    # 筛选事件索引
    if perc > 50:
        mask = data > thresholds
    else:
        mask = data < thresholds
    event_indices = np.where(mask)
    
    # 按格点整理事件
    events = [[] for _ in range(data.shape[1])]
    for t_idx, grid_idx in zip(*event_indices):
        events[grid_idx].append(t_idx)
    
    # 处理连续事件
    def remove_consecutive(arr):
        if len(arr) < 2:
            return arr
        diff = np.diff(arr)
        mask = np.concatenate([[True], diff != 1])
        return arr[mask]
    
    processed_events = [remove_consecutive(np.array(e)) for e in events]
    
    # 计算最大事件数并填充0
    max_events = max(len(e) for e in processed_events)
    ev_nb = np.array([np.pad(e, (0, max_events - len(e))) for e in processed_events], dtype=np.int16)
    nob = np.array([len(e) for e in processed_events], dtype=np.int16)
    
    return thresholds, events, ev_nb, nob

# 处理两个分位数
for quantile in [97.5, 2.5]:
    print(f"Processing {quantile}th percentile...")
    th, events, ev_nb, nob = process_quantile(globe_flat, quantile)
    
    # 保存结果
    np.save(f"./deseason/global_wd_score_{int(quantile)}_.npy", th)
    np.save(f"./deseason/global_wd_events_{int(quantile)}_.npy", np.array(events, dtype=object))
    np.save("./deseason/global_wd_bursts_cor_tas_perc%d" % quantile, ev_nb)
    np.save('./deseason/global_wd_nob_cor_tas_perc%d' % quantile, nob)


# def process_temperature_data(globe_diff):
#     num_days, num_lat, num_lon = globe_diff.shape
#     num_days = 393
#     num_grid_points = num_lat * num_lon
#     # 初始化 ev_nb 和 nob
#     ev_nb = np.zeros((2, num_grid_points, num_days), dtype=int)
#     nob = np.zeros((2, num_grid_points), dtype=int)

#     # 遍历每个格点
#     for lat in range(num_lat):
#         for lon in range(num_lon):
#             grid_point_index = lat * num_lon + lon
#             daily_data = globe_diff[:, lat, lon]
#             # 计算前2.5%和后2.5%分位数
#             lower_quantile = np.quantile(daily_data, 0.025)
#             upper_quantile = np.quantile(daily_data, 0.975)

#             # 处理大于前2.5%分位数的数据
#             upper_extreme_indices = np.where(daily_data > upper_quantile)[0]
#             # 删除连续的天数，只保留第一个
#             non_consecutive_upper_indices = []
#             if len(upper_extreme_indices) > 0:
#                 non_consecutive_upper_indices.append(upper_extreme_indices[0])
#                 for i in range(1, len(upper_extreme_indices)):
#                     if upper_extreme_indices[i] != upper_extreme_indices[i - 1] + 1:
#                         non_consecutive_upper_indices.append(upper_extreme_indices[i])
#             # 保存索引到 ev_nb
#             ev_nb[0, grid_point_index, :len(non_consecutive_upper_indices)] = non_consecutive_upper_indices
#             # 保存次数到 nob
#             nob[0, grid_point_index] = len(non_consecutive_upper_indices)

#             # 处理小于后2.5%分位数的数据
#             lower_extreme_indices = np.where(daily_data < lower_quantile)[0]
#             # 删除连续的天数，只保留第一个
#             non_consecutive_lower_indices = []
#             if len(lower_extreme_indices) > 0:
#                 non_consecutive_lower_indices.append(lower_extreme_indices[0])
#                 for i in range(1, len(lower_extreme_indices)):
#                     if lower_extreme_indices[i] != lower_extreme_indices[i - 1] + 1:
#                         non_consecutive_lower_indices.append(lower_extreme_indices[i])
#             # 保存索引到 ev_nb
#             ev_nb[1, grid_point_index, :len(non_consecutive_lower_indices)] = non_consecutive_lower_indices
#             # 保存次数到 nob
#             nob[1, grid_point_index] = len(non_consecutive_lower_indices)

#     return ev_nb, nob

# globe_tas = np.load("./origin_data.npy").squeeze().reshape([15695, 73, 144])
# globe_diff = np.diff(globe_tas, axis=0)
# ev_nb, nob = process_temperature_data(globe_diff)
# np.save(f"./origin/global_wd_bursts_97_1.npy", ev_nb[0])
# np.save(f"./origin/global_wd_nob_97_1.npy", nob[0])
# np.save(f"./origin/global_wd_bursts_2_1.npy", ev_nb[0])
# np.save(f"./origin/global_wd_nob_2_1.npy", nob[0])

# print("ev_nb shape:", ev_nb.shape)
# print("nob shape:", nob.shape)

print("Processing complete!")