import numpy as np
import scipy.stats as st
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing



index_day = np.arange(0, 15695, 1)
tlen = 15695
mnoe = index_day.shape[0] * (1 - 97.5 / 100.)
mnoe = int(mnoe) + 1
nodes = 2
y = 43
iteration_num = 2000  # paper parameter is 2000


@jit(nopython=True)
def EvSync_jit(dat, tlen, nodes, tm):
    """Calculates triangle of Event Synchronization Matrix Q as list"""
    # workspace (tmp event series)
    ex = np.empty(tlen, dtype=np.int32)
    ey = np.empty(tlen, dtype=np.int32)
    # output
    Q = 0
    # delay tau < taumax
    taumax = tm
    for i in range(nodes):
        lx = 0
        for t in range(tlen):
            if dat[i, t]:
                ex[lx] = t
                lx += 1
        for k in range(0, i):
            ly = 0
            for t_ in range(tlen):
                if dat[k, t_]:
                    ey[ly] = t_
                    ly += 1
            # count event synchronisations
            count = 0
            for m in range(1, lx - 1):
                for n in range(1, ly - 1):
                    dst = ex[m] - ey[n]
                    if dst > taumax:
                        continue
                    tmp = ex[m + 1] - ex[m]
                    if tmp > ex[m] - ex[m - 1]:
                        tmp = ex[m] - ex[m - 1]
                    tau = ey[n + 1] - ey[n]
                    if tau > ey[n] - ey[n - 1]:
                        tau = ey[n] - ey[n - 1]
                    if tau > tmp:
                        tau = tmp
                    tau //= 2
                    if abs(dst) <= taumax and abs(dst) < tau:
                        count += 1
                    if dst < -taumax:
                        break
            if lx < 3:
                q = 0
            elif ly < 3:
                q = 0
            else:
                q = count
            Q = q
    return Q


def EvSync(dat, tlen, nodes, tm):
    flat_dat = dat
    return EvSync_jit(flat_dat, tlen, nodes, tm)


def calculate_thresholds(i, j, tlen, nodes, tm, iteration_num):
    l = index_day.shape
    dayseries1 = np.zeros(l, dtype="bool")
    dayseries2 = np.zeros(l, dtype="bool")
    index_seas = np.array(index_day, dtype='int32')
    dayseries1[:i] = 1
    dayseries2[:j] = 1
    dat = np.zeros((nodes, tlen), dtype="bool")
    cor = np.zeros(iteration_num)
    for k in range(iteration_num):
        dat[0, index_seas] = np.random.permutation(dayseries1)
        dat[1, index_seas] = np.random.permutation(dayseries2)
        cor[k] = EvSync(dat, tlen, nodes, tm)
    th05 = st.scoreatpercentile(cor, 95)
    th02 = st.scoreatpercentile(cor, 98)
    th01 = st.scoreatpercentile(cor, 99)
    th005 = st.scoreatpercentile(cor, 99.5)
    th001 = st.scoreatpercentile(cor, 99.9)
    # print(i, j, th005)
    return i, j, th05, th02, th01, th005, th001

def call_calculate_thresholds(args):
    return calculate_thresholds(*args)

P1 = np.zeros((76636, 3))
P2 = np.zeros((76636, 3))
P3 = np.zeros((76636, 3))
P4 = np.zeros((76636, 3))
P5 = np.zeros((76636, 3))

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 仅在 Windows 上需要
    for tm in [30]:
        a = 0
        tasks = []
        for i in range(3, mnoe + 1):
            for j in range(3, i + 1):
                tasks.append((i, j, tlen, nodes, tm, iteration_num))

        num_processes = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                results = executor.map(call_calculate_thresholds, tasks)
                for i, j, th05, th02, th01, th005, th001 in results:
                    P1[a] = [i, j, th05]
                    P2[a] = [i, j, th02]
                    P3[a] = [i, j, th01]
                    P4[a] = [i, j, th005]
                    P5[a] = [i, j, th001]
                    a += 1
            except Exception as e:
                print(f"An error occurred: {e}")

        np.save('P_2000_3ms_mnoe393_thresholds_05_tm%d_jit' % tm, P1)
        np.save('P_2000_3ms_mnoe393_thresholds_02_tm%d_jit' % tm, P2)
        np.save('P_2000_3ms_mnoe393_thresholds_01_tm%d_jit' % tm, P3)
        np.save('P_2000_3ms_mnoe393_thresholds_005_tm%d_jit' % tm, P4)
        np.save('P_2000_3ms_mnoe393_thresholds_001_tm%d_jit' % tm, P5)
        