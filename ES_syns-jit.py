import sys
import numpy as np
import scipy.stats as st
import time
from numba import jit, prange



@jit(nopython=True, parallel=True)
def EvSync_jit(e, nodes, taumax, lookup_table, noep):
    l = int(nodes * (nodes - 1) // 2)
    Q = np.zeros(l, dtype=np.int32)
    c = 0
    for i in prange(nodes):
        for k in prange(i):
            count = 0
            for m in range(1, noep[i] - 1):
                if e[i, m] > 0:
                    for n in range(1, noep[k] - 1):
                        if e[k, n] > 0:
                            dst = e[i, m] - e[k, n]
                            
                            if dst > taumax:
                                continue
                            tmp = e[i, m + 1] - e[i, m]
                            if tmp > e[i, m] - e[i, m - 1]:
                                tmp = e[i, m] - e[i, m - 1]
                            tau = e[k, n + 1] - e[k, n]
                            if tau > e[k, n] - e[k, n - 1]:
                                tau = e[k, n] - e[k, n - 1]
                            if tau > tmp:
                                tau = tmp
                            tau //= 2
                            # print(noep[i], noep[k], e[i, m], e[k, n], dst, tau)
                            if abs(dst) <= taumax and abs(dst) < tau:
                                count += 1
                            if dst < -taumax:
                                break

            q = count
            e1 = noep[i]
            e2 = noep[k]
            if e1 < e2:
                e1, e2 = e2, e1
            # 直接从查找表中获取阈值
            threshold = lookup_table[e1, e2]
            condition = (threshold != -1 and q > threshold)
            # print(i,k,e1,e2,q,threshold,condition)
            if condition:
                local_c = int(i * (i - 1) // 2 + k)
                Q[local_c] = q
            else:
                local_c = int(i * (i - 1) // 2 + k)
                Q[local_c] = 0
    return Q


def EvSync(e, nodes, perc, tm, P, noep):
    print("Running calculation...")
    max_noep = 393
    print(max_noep)
    lookup_table = np.full((max_noep + 1, max_noep + 1), -1, dtype=np.int32)
    for row in range(P.shape[0]):
        e1 = int(P[row, 0])
        e2 = int(P[row, 1])
        threshold = int(P[row, 2])
        # print(e1,e2)
        lookup_table[e1, e2] = threshold
    Q = EvSync_jit(e, nodes, tm, lookup_table, noep)
    del e
    np.save(f'./deseason/global_wd_perc{perc}_tm{tm}_nb_sig005_jit', Q)
    print("Result saved.")
    del Q
    return 0


def master():
    for perc in [97,2]:
        print("Percentile:", perc)
        for tm in [30]:
            print("Time threshold:", tm)
            dat = np.load('./deseason/global_wd_bursts_cor_tas_perc%d.npy' % perc).astype((np.int32))
            # data_3d = data.reshape(73, 144, 393)
            # latitudes = np.linspace(90, -90, 73)

            # north_50_index = np.argmin(np.abs(latitudes - 50))
            # south_50_index = np.argmin(np.abs(latitudes + 50))
            # print(north_50_index,south_50_index)
            # subset_3d = data_3d[north_50_index:south_50_index + 1, :, :]
            # dat = subset_3d.reshape(-1, 393)
            P = np.load('P_2000_3ms_mnoe393_thresholds_005_tm%d_jit.npy' % tm).astype((np.int32))
            noep = np.load('./deseason/global_wd_nob_cor_tas_perc%d.npy' % perc).astype((np.int32))
            
            # noep_2d = noep1.reshape(73, 144)
            # noep1 = noep_2d[north_50_index:south_50_index + 1, :]
            # noep = noep1.reshape(-1)
            nodes = dat.shape[0]
            noe = dat.shape[1]
            print(dat.shape)
            print(noep.shape)
            # dat = dat.reshape(nodes * noe)
            print("Calculating event synchronization...")
            EvSync(dat, nodes, perc, tm, P, noep)
            del dat


master()
    