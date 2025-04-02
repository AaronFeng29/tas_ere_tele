import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.colors as mcolors
from scipy.interpolate import griddata

def latlon_to_cartesian(lat, lon, R):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)
    return x, y, z


def spherical_distance(p1, p2, R):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dot = x1 * x2 + y1 * y2 + z1 * z2
    cos_theta = dot / (R * R)
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # 防止浮点误差
    theta = np.arccos(cos_theta)
    return R * theta

def index_in_lat_s50_n50():
    atitudes = np.linspace(90, -90, 73)
    # 找到南纬50度和北纬50度对应的索引
    lat_50_index = np.where(latitudes <= 50)[0][0]
    lat_minus_50_index = np.where(latitudes <= -50)[0][0]
    print(lat_50_index,lat_minus_50_index)
    # 对一维数组进行切片
    # 注意：因为数组是按照纬度和经度排列的，每个纬度有144个经度值
    start_index = lat_50_index * 144
    end_index = lat_minus_50_index * 144
    return start_index, end_index




def cal_deg_distance(adj_matrix):
    start_index, end_index = index_in_lat_s50_n50()
    selected_distances2 = []
    selected_distances7 = []
    distances = []
    n50s50_distances = []
    
    lat_diff = []
    selected_lat_diff2 = []
    selected_lat_diff7 = []
    n50s50_lat_diff = []
    
    deg = np.zeros(n)
    length = deg[start_index: end_index].shape    
    selected_deg7 = np.zeros(selected_index7.shape)
    selected_deg2 = np.zeros(selected_index2.shape)
    n50s50_deg = np.zeros(length)

    for i in range(n):
        for k in range(i):
            index = i * (i - 1) // 2 + k
            # adj_matrix是下三角邻接矩阵
            if adj_matrix[index] > 0:
                deg[i] += 1
                deg[k] += 1
                p1 = latlon_to_cartesian(*points[i], R)
                p2 = latlon_to_cartesian(*points[k], R)   
                p1lat = int(i / 144)
                p2lat = int(k / 144)
                end_lat = latitudes[p2lat]
                start_lat = latitudes[p1lat]
                lat_diff.append(end_lat - start_lat)
                dist = spherical_distance(p1, p2, R)
                distances.append(dist)
                if start_index <= i < end_index and start_index <= k < end_index:
                    n50s50_distances.append(dist)
                    n50s50_deg[i-start_index] += 1
                    n50s50_deg[k-start_index] += 1
                    n50s50_lat_diff.append(end_lat - start_lat)
                if i in selected_index7 and k in selected_index7:
                    index_1 = np.where(selected_index7 == i)[0][0]
                    index_2 = np.where(selected_index7 == k)[0][0]
                    selected_deg7[index_1] += 1
                    selected_deg7[index_2] += 1
                    selected_distances7.append(dist)
                    selected_lat_diff7.append(end_lat - start_lat)
                if i in selected_index2 and k in selected_index2:
                    index_1 = np.where(selected_index2 == i)[0][0]
                    index_2 = np.where(selected_index2 == k)[0][0]
                    selected_distances7.append(dist)
                    selected_deg2[index_1] += 1
                    selected_deg2[index_2] += 1
                    selected_lat_diff2.append(end_lat - start_lat)
                    selected_distances2.append(dist)
    distances = np.array(distances)
    # distances = distances[distances>0]
    selected_distances2 = np.array(selected_distances2)
    selected_distances7 = np.array(selected_distances7)
    n50s50_distances = np.array(n50s50_distances)
    np.save(f'{data}/deg_selected6570_perc{perc}_tm30_nb_sig005_jit.npy', selected_deg2)
    np.save(f'{data}/dist_selected6570_perc{perc}_tm30_nb_sig005_jit.npy', selected_distances2)
    np.save(f'{data}/latdiff_selected6570_perc{perc}_tm30_nb_sig005_jit.npy', selected_lat_diff2)
    np.save(f'{data}/deg_selected726_perc{perc}_tm30_nb_sig005_jit.npy', selected_deg7)
    np.save(f'{data}/dist_selected726_perc{perc}_tm30_nb_sig005_jit.npy', selected_distances7)
    np.save(f'{data}/latdiff_selected726_perc{perc}_tm30_nb_sig005_jit.npy', selected_lat_diff7)
    np.save(f'{data}/deg_n50s50_perc{perc}_tm30_nb_sig005_jit.npy', n50s50_deg)
    np.save(f'{data}/dist_n50s50_perc{perc}_tm30_nb_sig005_jit.npy', n50s50_distances)
    np.save(f'{data}/latdiff_n50s50_perc{perc}_tm30_nb_sig005_jit.npy', n50s50_lat_diff)
    np.save(f'{data}/deg_perc{perc}_tm30_nb_sig005_jit.npy', deg)
    np.save(f'{data}/dist_perc{perc}_tm30_nb_sig005_jit.npy', distances)
    np.save(f'{data}/latdiff_perc{perc}_tm30_nb_sig005_jit.npy', lat_diff)
    return n50s50_distances, selected_distances7, selected_distances2, distances

def cal_deg_distance_dirc(adj_matrix, perc):
    adj_matrix = adj_matrix.reshape([10512,10512])
    start_index, end_index = index_in_lat_s50_n50()
    selected_distances2 = []
    selected_distances7 = []
    n50s50_distances = []
    
    in_deg = np.zeros(10512)
    out_deg = np.zeros(10512)
    length = in_deg[start_index: end_index].shape    
    selected_in_deg7 = np.zeros(selected_index7.shape)
    selected_in_deg2 = np.zeros(selected_index2.shape)
    selected_out_deg7 = np.zeros(selected_index7.shape)
    selected_out_deg2 = np.zeros(selected_index2.shape)
    n50s50_in_deg = np.zeros(length)
    n50s50_out_deg = np.zeros(length)

    # 计算所有连边的距离
    distances = []
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            if adj_matrix[i,k] > 0:
                p1 = latlon_to_cartesian(*points[i], R)
                p2 = latlon_to_cartesian(*points[k], R)
                dist = spherical_distance(p1, p2, R)
                out_deg[i] += 1
                in_deg[k] += 1
                distances.append(dist)
                if start_index <= i < end_index and start_index <= k < end_index:
                    n50s50_out_deg[i-start_index] += 1
                    n50s50_in_deg[k-start_index] += 1
                    n50s50_distances.append(dist)
                if i in selected_index7 and k in selected_index7:
                    index_1 = np.where(selected_index7 == i)[0][0]
                    index_2 = np.where(selected_index7 == k)[0][0]
                    selected_out_deg7[index_1] += 1
                    selected_in_deg7[index_2] += 1
                    selected_distances7.append(dist)
                    
                if i in selected_index2 and k in selected_index2:
                    index_1 = np.where(selected_index2 == i)[0][0]
                    index_2 = np.where(selected_index2 == k)[0][0]
                    selected_out_deg2[index_1] += 1
                    selected_in_deg2[index_2] += 1
                    selected_distances2.append(dist)
    distances = np.array(distances)
    distances = distances[distances>0]
    n50s50_distances = np.array(n50s50_distances)
    selected_distances7 = np.array(selected_distances7)
    selected_distances2 = np.array(selected_distances2)
    np.save(f'{data}/dist_perc{perc}_tm30_nb_sig005_jit_dirc.npy',distances)
    np.save(f'{data}/in_deg_perc{perc}_tm30_nb_sig005_jit.npy', in_deg)
    np.save(f'{data}/out_deg_perc{perc}_tm30_nb_sig005_jit.npy', out_deg)
   
    np.save(f'{data}/dist_selected6570_perc{perc}_tm30_nb_sig005_jit_dirc.npy', selected_distances2)
    np.save(f'{data}/in_deg_selected6570_perc{perc}_tm30_nb_sig005_jit.npy', selected_in_deg2)
    np.save(f'{data}/out_deg_selected6570_perc{perc}_tm30_nb_sig005_jit.npy', selected_out_deg2)
    
    np.save(f'{data}/dist_selected726_perc{perc}_tm30_nb_sig005_jit_dirc.npy', selected_distances7)
    np.save(f'{data}/in_deg_selected726_perc{perc}_tm30_nb_sig005_jit.npy', selected_in_deg7)
    np.save(f'{data}/out_deg_selected726_perc{perc}_tm30_nb_sig005_jit.npy', selected_out_deg7)
    
    np.save(f'{data}/dist_n50s50_perc{perc}_tm30_nb_sig005_jit_dirc.npy', n50s50_distances)
    np.save(f'{data}/in_deg_n50s50_perc{perc}_tm30_nb_sig005_jit.npy', n50s50_in_deg)
    np.save(f'{data}/out_deg_n50s50_perc{perc}_tm30_nb_sig005_jit.npy', n50s50_out_deg)
    
    return in_deg, out_deg, n50s50_distances, selected_distances7, selected_distances2, distances

def cal_deg_distance_pn(adj_matrix1, adj_matrix2):
    adj_matrix1 = adj_matrix1.reshape([10512,10512])
    adj_matrix2 = adj_matrix2.reshape([10512,10512])
    start_index, end_index = index_in_lat_s50_n50()
    selected_distances2 = []
    selected_distances7 = []
    n50s50_distances = []
    marker = np.zeros([10512,10512])
    deg = np.zeros(10512)
    length = deg[start_index: end_index].shape    
    selected_deg7 = np.zeros(selected_index7.shape)
    selected_deg2 = np.zeros(selected_index2.shape)
    n50s50_deg = np.zeros(length)
    in_deg = np.zeros(10512)
    out_deg = np.zeros(10512)
    lat_diff = []
    selected_lat_diff2 = []
    selected_lat_diff7 = []
    n50s50_lat_diff = []
    
    # 计算所有连边的距离
    distances = []
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            if adj_matrix1[i, k] > 0 or adj_matrix2[i, k]:
                p1 = latlon_to_cartesian(*points[i], R)
                p2 = latlon_to_cartesian(*points[k], R)
                dist = spherical_distance(p1, p2, R)
                out_deg[i] += 1
                in_deg[k] += 1
                if i > k:
                    marker[i, k] = 1
                if marker[k, i] > 0:
                    continue
                else:
                    p1lat = int(i / 144)
                    p2lat = int(k / 144)
                    end_lat = latitudes[p2lat]
                    start_lat = latitudes[p1lat]
                    if dist>10:
                        lat_diff.append(end_lat - start_lat)
                    deg[i] += 1
                    deg[k] += 1
                    distances.append(dist)
                    if start_index <= i < end_index and start_index <= k < end_index:
                        n50s50_deg[i-start_index] += 1
                        n50s50_distances.append(dist)
                        n50s50_lat_diff.append(end_lat - start_lat)
                    if i in selected_index7 and k in selected_index7:
                        index_1 = np.where(selected_index7 == i)[0][0]
                        index_2 = np.where(selected_index7 == k)[0][0]
                        selected_deg7[index_1] += 1
                        selected_distances7.append(dist)
                        selected_lat_diff7.append(end_lat - start_lat)
                    if i in selected_index2 and k in selected_index2:
                        index_1 = np.where(selected_index2 == i)[0][0]
                        index_2 = np.where(selected_index2 == k)[0][0]
                        selected_deg2[index_1] += 1
                        selected_distances2.append(dist)
                        selected_lat_diff2.append(end_lat - start_lat)
    distances = np.array(distances)
    distances = distances[distances>0]
    n50s50_distances = np.array(n50s50_distances)
    selected_distances7 = np.array(selected_distances7)
    selected_distances2 = np.array(selected_distances2)
    # np.save(f'{data}/dist_pn_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy',distances)
    # np.save(f'{data}/deg_pn_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', deg)
    # np.save(f'{data}/latdiff_pn_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', lat_diff)
    
    # np.save(f'{data}/dist_pn_selected6570_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_distances2)
    # np.save(f'{data}/deg_pn_selected6570_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_deg2)
    # np.save(f'{data}/latdiff_selected6570_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_lat_diff2)
    
    # np.save(f'{data}/dist_pn_selected726_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_distances7)
    # np.save(f'{data}/deg_pn_selected726_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_deg7)
    # np.save(f'{data}/latdiff_selected726_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', selected_lat_diff7)
    
    # np.save(f'{data}/dist_pn_n50s50_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', n50s50_distances)
    # np.save(f'{data}/deg_pn_n50s50_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', n50s50_deg)
    # np.save(f'{data}/latdiff_pn_n50s50_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', n50s50_lat_diff)
    
    np.save(f'{data}/in_deg_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', in_deg)
    np.save(f'{data}/out_deg_perc{perc1}-{perc2}_tm30_nb_sig005_jit.npy', out_deg)
    
    return n50s50_distances, selected_distances7, selected_distances2, distances

def cu_deg_nlist(adj_matrix):
    num_nodes = n
    deg = np.zeros(num_nodes, dtype='int32')
    # 遍历邻接矩阵，计算每个节点的度数
    for i in range(num_nodes):
        for k in range(i):
            index = i * (i - 1) // 2 + k
            if adj_matrix[index] > 0:
                deg[i] += 1
                deg[k] += 1
    
    # 计算累积度数数组 cu_deg
    cu_deg = np.zeros(num_nodes + 1, dtype='uint64')
    s = 0
    for i in range(num_nodes + 1):
        cu_deg[i] = s
        if i == num_nodes:
            continue
        s += deg[i]
        # cu_deg0保存0 1保存节点0的度 2保存节点0+1的度 3保存节点0+1+2的度
    
    # 初始化邻接列表 nlist
    nlist = np.zeros(np.sum(deg), dtype='int32')
    
    # 初始化位置数组
    positions = cu_deg.copy()
    
    # 遍历邻接矩阵，填充邻接列表 nlist
    for i in range(num_nodes):
        for k in range(i):
            index = i * (i - 1) // 2 + k
            if adj_matrix[index] > 0:
                v = positions[k]
                w = positions[i]
                nlist[v] = i
                nlist[w] = k
                positions[k] += 1
                positions[i] += 1
    np.save(f'{data}/cu_deg_perc{perc}_tm30_nb_sig005_jit.npy', cu_deg)
    np.save(f'{data}/nlist_perc{perc}_tm30_nb_sig005_jit.npy', nlist)
    return cu_deg, nlist

def draw_pdf(distances):

    distances = distances[distances>0]
    # 计算距离的 PDF
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    bins = np.logspace(np.log10(min_dist), np.log10(max_dist), num=40)
    # counts, bin_edges = np.histogram(distances, bins=bins, density = True)
    
    loghist = np.histogram(distances, bins=bins, density = True)
    logx1 = loghist[1][:-1] + (loghist[1][1:] - loghist[1][:-1]) / 2.
    # np.save(f'{data}/dist_pdf_perc{perc1}-{perc2}_tm30.npy',loghist[0])
    # np.save(f'{data}/dist_pdf_logx_perc{perc1}-{perc2}_tm30.npy',logx1)
    # bin_widths = np.diff(bin_edges)
    # pdf = counts / (len(distances) * bin_widths)
    # bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    
    # 绘制双对数 PDF 图
    plt.figure(figsize=(9, 3))
    plt.loglog(logx1, loghist[0], color='r', ls='None', marker='o', markeredgewidth=0.5, alpha=.9, markersize=10, fillstyle='none')
    # plt.loglog(bin_centers, pdf, color='r', ls='None', marker='o', markeredgewidth=0.5, alpha=.9, markersize=10, fillstyle='none')
    plt.xlabel('Distance (km)')
    plt.ylabel('Probability Density')
    plt.title('Log-Log Plot of Distance Distribution')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.show()
    
def draw_deg(deg, lon, lat):
    lon[-1] = 360
    dmax = 2000
    
    Lon, Lat = np.meshgrid(lon, lat)
    data = deg.reshape([73, 144])/dmax
    
    # 创建地图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    
    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    vmin = 0.01
    vmax = 0.8
    levels = np.linspace(vmin, vmax, 20)    
    # 绘制数据
    im = ax.contourf(Lon, Lat, data, levels=levels, cmap='viridis', transform=ccrs.PlateCarree(), extend='both')
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='deg', extend='both')
    # 设置标题
    ax.set_title('')
    plt.show()
    
def draw_n50s50_deg(deg, lon, lat):
    start_index, end_index = index_in_lat_s50_n50()
    lon[-1] = 360
    dmax = 1000
    lat = lat[16:56]
    Lon, Lat = np.meshgrid(lon, lat)
    lat_len = int(deg.shape[0] / 144)
    print(deg.shape[0], lat_len)
    data = deg.reshape([lat_len, 144]) / dmax

    
    # 创建地图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    
    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    vmin = 0.01
    vmax = 0.8
    levels = np.linspace(vmin, vmax, 20)    
    # 绘制数据
    im = ax.contourf(Lon, Lat, data, levels=levels, cmap='viridis', transform=ccrs.PlateCarree(), extend='both')
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='deg', extend='both')
    # 设置标题
    ax.set_title('')
    plt.show()
    
def draw_divergence(in_deg, out_deg, lon, lat):
    lon[-1] = 360.
    dmax = 200
    deg = out_deg - in_deg
    print('min:', int(np.min(deg)), 'max:', int(np.max(deg)))
    Lon, Lat = np.meshgrid(lon, lat)
    data = deg.reshape([73, 144])/dmax
    
    # 创建地图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    
    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    vmin = -1
    vmax = 1
    levels = np.linspace(vmin, vmax, 20)
    # 绘制数据
    im = ax.contourf(Lon, Lat, data, levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(), extend='both')
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='deg', extend='both')
    # 设置标题
    ax.set_title('')
    plt.show()    
    
def draw_selected_deg(deg, lon, lat):
    full_data = np.full(73 * 144, np.nan)
    count = 0
    for idx in selected_index2:
        full_data[idx] = deg[count]
        count+=1
    lon[-1] = 360
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    full_data = full_data.reshape([73,144])
    
    if np.any(np.abs(lon - 0) < 1e-5):
        # 复制最后一列（360°）到第一列（0°）
        full_data[:, 0] = full_data[:, -1].copy()

    
    valid = ~np.isnan(full_data)
    valid_lats = lat_grid[valid]
    valid_lons = lon_grid[valid]
    valid_vals = full_data[valid]

    # 创建插值网格（使用原网格）
    interpolated = griddata(
        (valid_lats.ravel(), valid_lons.ravel()),
        valid_vals.ravel(),
        (lat_grid, lon_grid),
        method='linear'
    )
    
    # 创建地图
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    
    # 添加地理特征
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    vmin = 0
    vmax = 150
    levels = np.linspace(vmin, vmax, 10)
    norm = mcolors.BoundaryNorm(levels, plt.cm.viridis.N)
    # 绘制数据
    im = ax.contourf(lon_grid, lat_grid, interpolated, levels=levels, cmap='viridis', transform=ccrs.PlateCarree(), extend='both')
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', label='deg', extend='both')
    # 设置标题
    ax.set_title('')
    plt.show()
    
# 地球半径（公里）
R = 6371
tm = 30
# 点数
n = 10512 # n50s50:5904
perc = 97
perc1 = 97
perc2 = 2
data = 'origin' # 'deseason

indexarr2 = []
f = open('../Resolution2.5')
line = f.readline().strip()
node_index = line.split('	')
indexarr2.append(node_index[0])
while line:
    line = f.readline().strip()
    node_index = line.split('	')
    indexarr2.append(node_index[0])
f.close()
selected_index2 = np.array(indexarr2[:-1]).astype('int32')

indexarr7 = []
f = open('../Resolution7.5')
line = f.readline().strip()
node_index = line.split('	')
indexarr7.append(node_index[0])
while line:
    line = f.readline().strip()
    node_index = line.split('	')
    indexarr7.append(node_index[0])
f.close()
selected_index7 = np.array(indexarr7[:-1]).astype('int32')




# adj_matrix = np.load(f'./{data}/global_wd_perc{perc}_tm30_nb_sig005_jit.npy')
adj_matrix = np.load(f'./{data}/global_wd_perc{perc1}_tm{tm}_nb_sig005_jit_dirc_3.npy').reshape([10512,10512])
adj_matrix_t = np.load(f'./{data}/global_wd_perc{perc2}_tm30_nb_sig005_jit_dirc_3.npy').reshape([10512,10512])
dat = np.load(f'./{data}/global_wd_bursts_cor_tas_perc{perc1}.npy')
print(np.allclose(adj_matrix, adj_matrix.T))

latitudes = np.linspace(90, -90, 73)
longitudes = np.linspace(0, 360, 144, endpoint=False)
latitudes1 = np.load('../lat.npy')
longitudes1 = np.load('../lon.npy')
points = []
for lat in latitudes:
    for lon in longitudes:
        points.append((lat, lon))
points = np.array(points)
cal_deg_distance_dirc(adj_matrix, perc1)
in_deg, out_deg, n50s50_distances, selected_distances7, selected_distances2, distances = cal_deg_distance_dirc(adj_matrix_t, perc2)
