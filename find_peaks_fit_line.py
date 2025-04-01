# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:04:59 2024

@author: Aaron
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as basemap
from scipy.stats import kstest
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def loghist(distances):
    distances = distances[distances>10]
    # 计算距离的 PDF
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    bins = np.logspace(np.log10(min_dist), np.log10(max_dist), num=40)
    # counts, bin_edges = np.histogram(distances, bins=bins, density = True)

    loghist = np.histogram(distances, bins=bins, density = True)
    logx1 = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.
    return loghist, logx1

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def theoretical_cdf(x, alpha, xmin):
    return 1 - (xmin / x)**(alpha - 1)

def ks_test(data, alpha, xmin):
    data = data[data >= xmin]  # 只考虑大于等于 xmin 的数据
    ecdf_x, ecdf_y = ecdf(data)
    tcdf_y = theoretical_cdf(ecdf_x, alpha, xmin)
    ks_stat, p_value = kstest(ecdf_y, lambda y: theoretical_cdf(ecdf_x, alpha, xmin)[np.searchsorted(ecdf_x, ecdf_x)])
    return ks_stat, p_value

def mle_power_law(data, xmin):
    data = data[data >= xmin]
    alpha = 1 + len(data) / np.sum(np.log(data / xmin))
    return alpha
perc1 = 97
perc2 = 2
tm = 30
frac = 0.001
perc = perc1
data = "deseason"
pst_distances = np.load(f'./{data}/dist_perc{perc1}_tm30_nb_sig005_jit.npy')
ngt_distances = np.load(f'./{data}/dist_perc{perc2}_tm30_nb_sig005_jit.npy')
pst_ngt_distances = np.load(f'./{data}/dist_pn_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy')
# pst_distances = np.load(f'origin/dist_n50s50_perc{perc1}_tm30_nb_sig005_jit.npy')
# ngt_distances = np.load(f'origin/dist_n50s50_perc{perc2}_tm30_nb_sig005_jit.npy')
# pst_ngt_distances = np.load(f'origin/dist_pn_n50s50_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy')
# pst_distances = np.load(f'origin/dist_selected6570_perc{perc1}_tm30_nb_sig005_jit.npy')
# ngt_distances = np.load(f'origin/dist_selected6570_perc{perc2}_tm30_nb_sig005_jit.npy')
# pst_ngt_distances = np.load(f'origin/dist_pn_selected6570_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy')
loghist1, logx1 = loghist(pst_distances)
loghist2, logx2 = loghist(ngt_distances)
loghist3, logx3 = loghist(pst_ngt_distances)

loghist1 = loghist3[0]
logx1 = logx3
mode = 'pst_ngt'
print(loghist1.shape)
# print(logx1)


def power_law(x, a, b):
    return a * x**(-b)


# 保存下降速度降为零的位置
zeros = []


plt.figure(figsize=(9, 3))
# 找到数据中的峰
peaks1, _ = find_peaks(loghist1, height=0)
valleys1, _ = find_peaks(-loghist1, height=None, threshold=None, distance=1)
valleys1 = np.append(valleys1, loghist1.shape[0] - 1)
peaks_valleys = np.sort(np.concatenate((peaks1, valleys1)))
print(peaks1)
print(valleys1)
print(peaks_valleys)

for i in range(len(peaks_valleys)):
    start = peaks_valleys[i]
    if i == len(peaks_valleys) - 1:
        end = len(logx1)
    else:
        end = peaks_valleys[i+1]
    if start in peaks1 and end in valleys1:
        x_data = logx1[start:end+1]
        y_data = loghist1[start:end+1]
    else:
        continue

    if end - start < 2:
        continue
    print(start, end)
    # 尝试拟合power law
    try:
        # popt, _ = curve_fit(power_law, x_data, y_data, p0=[np.max(y_data), 0.5], bounds=([0, 0], [np.inf, 2]), maxfev=2000)
        popt, _ = curve_fit(power_law, x_data, y_data, p0=[np.max(y_data), 1], maxfev=2000)
        a, b = popt
        ks_stat, p_value = ks_test(y_data, b, min(y_data))
        # print('ks_stat1:', ks_stat, 'p:', p_value, 'alpha:', b)
        # alpha_estimated = mle_power_law(y_data, min(y_data))
        # ks_stat1, p_value1 = ks_test(y_data, alpha_estimated, min(y_data))
        # print('ks_stat2:', ks_stat1, 'p:', p_value1, 'alpha:', alpha_estimated)
        # y1 = theoretical_cdf(y_data, alpha_estimated, min(y_data))
        # plt.loglog(x_data, y1, '--', label=r'$\alpha_1 = %.2f$' % (b))
        plt.loglog(x_data, power_law(x_data, a, b), '--', label=r'$\alpha_1 = %.2f$, %d %d' % (b, start, end))
        # 找到斜率接近零的位置
        for j in range(start, end):
            threshold = 1*1e-7
            if power_law(logx1[j], a, b) < threshold:
            #1*1e-7: 你可以调整这个阈值
                # print(logx1[j])
                zeros.append(logx1[j])
                break
    except RuntimeError:
        print("Fit did not converge for segment", i)




# 绘制结果

plt.loglog(logx1, loghist1, color='r', ls='None', marker='o', markeredgewidth=0.5, alpha=.9, markersize=10, fillstyle='none')


for peaks11 in peaks1:
    plt.axvline(x=logx1[peaks11], color='m', linestyle='--', alpha=0.7)
for valleys11 in valleys1:
    plt.axvline(x=logx1[valleys11], color='orange', linestyle='--', alpha=0.7)
# plt.ylim(10 ** -8, 10 ** -3)#(10 ** -9, 10 ** -2)
# plt.xlim(50, 25000)
plt.xlabel('Distance [km]')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.savefig(f'./pics/{data}_{mode}_distance.jpg', dpi=300, bbox_inches='tight')
plt.show()



