# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:32:45 2025

@author: Aaron
"""
import numpy as np
globe_tas_ = np.squeeze(np.load("./origin_data.npy")).reshape([15695,73*144])
year_num = 43
nodes_num = 73*144
day_num = 15695
temp = globe_tas_
count = 0
temp_year_list = []
for y in range(0, year_num):
    temp_h1 = 365 * y
    temp_year_list.append(temp_h1)
    count += 1

yearav = np.zeros((365, nodes_num))
for day in range(0, 365):
    temp_h1_list_new = [a + day for a in temp_year_list]  # temp_h1_list_new为所有年份的第day天
    yearav[day, :] = np.mean(temp[temp_h1_list_new, :], axis=0)

st = np.zeros((365, nodes_num))
for day in range(0, 365):
    temp_h1_list_new = [a + day for a in temp_year_list]
    st[day, :] = np.std(temp[temp_h1_list_new, :], axis=0)

temp_ = np.zeros((day_num, nodes_num))
for k in range(0, day_num):
    u = int(k % 365)
    temp_[k, :] = np.divide((temp[k, :] - yearav[u, :]), st[u, :])

np.save("deseason_data.npy",temp_)