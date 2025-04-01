import sys
import numpy as np
import scipy.stats as st
import time
from numba import jit, prange



@jit(nopython=True, parallel=True)
def EvSync_jit(event1, event2, nodes, taumax, lookup_table, noep1, noep2):
    l = nodes * nodes
    Q = np.zeros(l, dtype=np.int32)
    for i in prange(nodes):
        for k in prange(nodes):
            count = 0
            for m in range(1, noep1[i] - 1):
                if event1[i, m] > 0:
                    for n in range(1, noep2[k] - 1):
                        if event2[k, n] > 0:
                            dst1 = event1[i, m] - event2[k, n]
                            if dst1 > 0:
                                continue
                            tmp1 = event1[i, m + 1] - event1[i, m]
                            if tmp1 > event1[i, m] - event1[i, m - 1]:
                                tmp1 = event1[i, m] - event1[i, m - 1]
                            tau1 = event2[k, n + 1] - event2[k, n]
                            if tau1 > event2[k, n] - event2[k, n - 1]:
                                tau1 = event2[k, n] - event2[k, n - 1]
                            if tau1 > tmp1:
                                tau1 = tmp1
                            tau1 //= 2

                            # print(noep[i], noep[k], e[i, m], e[k, n], dst, tau)
                            if abs(dst1) <= taumax and abs(dst1) < tau1:
                                count += 1
                            if dst1 < -taumax:
                                break

            q = count
            e1 = noep1[i]
            e2 = noep2[k]
            if e1 < e2:
                e1, e2 = e2, e1
            # 直接从查找表中获取阈值
            threshold = lookup_table[e1, e2]
            condition = (threshold != -1 and q > threshold)
            # print(i,k,e1,e2,q,threshold,condition)
            if condition:
                local_c = int(i * nodes + k)
                Q[local_c] = q
            else:
                local_c = int(i * nodes + k)
                Q[local_c] = 0
    return Q


def EvSync(event1, event2, nodes, perc1, perc2, tm, P, noep1, noep2):
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
    Q = EvSync_jit(event1, event2, nodes, tm, lookup_table, noep1, noep2)
    np.save(f'./deseason/global_wd_perc{perc1}-{perc2}_tm{tm}_nb_sig005_jit_dirc', Q)
    print("Result saved.")
    del Q
    return 0


def master():
    for tm in [30]:
        perc1 = 97
        perc2 = 2
        print("Time threshold:", tm)
        datp = np.load('./deseason/global_wd_bursts_cor_tas_perc%d.npy' % perc1).astype((np.int32))
        datn = np.load('./deseason/global_wd_bursts_cor_tas_perc%d.npy' % perc2).astype((np.int32))
        # data_3d = data.reshape(73, 144, 393)
        # latitudes = np.linspace(90, -90, 73)

        # north_50_index = np.argmin(np.abs(latitudes - 50))
        # south_50_index = np.argmin(np.abs(latitudes + 50))
        # print(north_50_index,south_50_index)
        # subset_3d = data_3d[north_50_index:south_50_index + 1, :, :]
        # dat = subset_3d.reshape(-1, 393)
        P = np.load('P_2000_3ms_mnoe393_thresholds_005_tm%d_jit_dirc.npy' % tm).astype((np.int32))
        noep1 = np.load('./deseason/global_wd_nob_cor_tas_perc%d.npy' % perc1).astype((np.int32))
        noep2 = np.load('./deseason/global_wd_nob_cor_tas_perc%d.npy' % perc2).astype((np.int32))
        
        # noep_2d = noep1.reshape(73, 144)
        # noep1 = noep_2d[north_50_index:south_50_index + 1, :]
        # noep = noep1.reshape(-1)
        nodes = datp.shape[0]
        print(datp.shape)
        print(noep1.shape)
        # dat = dat.reshape(nodes * noe)
        print("Calculating event synchronization...")
        EvSync(datp, datn, nodes, perc1, perc2, tm, P, noep1, noep2)
        EvSync(datn, datp, nodes, perc2, perc1, tm, P, noep2, noep1)
        


master()
    