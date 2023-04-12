import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import sys
import pickle
import xlrd
from sklearn.cluster import KMeans
import random
import numpy.linalg as LA
"""预处理脚本"""
##! cstr包括身份，但不包括零
cstr_source = {  #* u-side 
   "LncRNA" : [0,2,3]
}

cstr_target = {  #* i-side
     "diseases" : [1,4,6]
}
"D(-1/2)AD(-1/2)HW"
"""对称地规范化邻接矩阵-D(-1/2)"""
def normalize_sym(adj):
    # 1.sum(axis=1)以后就是将一个矩阵的每一行向量相加(求度矩阵)
    rowsum = np.array(adj.sum(1))
    # flatten只能适用于numpy对象,即返回一个一维数组,默认是按行的方向降
    # 2.首先行向量和-0.5次方，然后返回一个一维数组
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # 用于检查数字是否为无穷大(正数或负数)，它接受数字，如果给定数字为正无穷大或负无穷大，则返回True。返回False。
    # 3.超出范围的赋值为0
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # 4.变成对角矩阵
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # 5.原矩阵点乘对角矩阵，再转置，点乘对角矩阵。
    # 返回稀疏矩阵的coo_matrix形式
    "D(-1/2)AD(-1/2)"
    return adj.dot(adj).dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

""""对稀疏矩阵进行(行)规范化处理-度矩阵"""
def normalize_row(mx):
    # 1.每行取和
    rowsum = np.array(mx.sum(1))
    # 2.每行和取倒数，返回一维数组
    r_inv = np.power(rowsum, -1).flatten()
    # 无效值赋值为0
    r_inv[np.isinf(r_inv)] = 0.
    # 4.变成对角矩阵
    r_mat_inv = sp.diags(r_inv)
    # 5.处理过后的对角矩阵与原来矩阵点乘
    "D(-1)A"
    mx = r_mat_inv.dot(mx).dot(mx)
    return mx.tocoo()

""""将scipy稀疏矩阵转换为Torch稀疏张量"""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



"""预处理 drug"""
def preprocess_Drug(prefix):
    # ld = pd.read_excel(os.path.join(prefix, "L-d.xlsx"), names=['lid', 'did', 'rating'])
    # # ldall = pd.read_excel(os.path.join(prefix, "lncrna-diseases-associationidx.xlsx"),names=['lid', 'did', 'rating'])
    # lm = pd.read_excel(os.path.join(prefix, "L-M.xlsx"), names=['lid', 'mid', 'weight'])
    # md = pd.read_excel(os.path.join(prefix, "m-d.xlsx"), names=['mid', 'did', 'weight'])
    # mm = pd.read_excel(os.path.join(prefix, "m-m03.xlsx"), names=['m1', 'm2', 'weight'])
    # dd = pd.read_excel(os.path.join(prefix, "d-d03.xlsx"), names=['d1', 'd2', 'weight']).drop_duplicates().reset_index(drop=True)  ## not full; sym
    # ll = pd.read_excel(os.path.join(prefix, "l-l03.xlsx"), names=['l1', 'l2', 'weight']).drop_duplicates().reset_index(drop=True)  ## not full; sym
    # l_num =240
    # d_num=412
    ld = pd.read_excel(os.path.join(prefix, "l-d.xlsx"), names=['lid', 'did', 'rating'])
    ldall = pd.read_excel(os.path.join(prefix, "l-dnegall.xlsx"),names=['lid', 'did', 'rating'])
    lm = pd.read_excel(os.path.join(prefix, "l-m.xlsx"), names=['lid', 'mid', 'weight'])
    md = pd.read_excel(os.path.join(prefix, "m-d.xlsx"), names=['mid', 'did', 'weight'])
    mm = pd.read_excel(os.path.join(prefix, "2m-m03.xlsx"), names=['m1', 'm2', 'weight'])
    dd = pd.read_excel(os.path.join(prefix, "2d-d03.xlsx"), names=['d1', 'd2', 'weight']).drop_duplicates().reset_index(drop=True)  ## not full; sym
    ll = pd.read_excel(os.path.join(prefix, "2l-l03.xlsx"), names=['l1', 'l2', 'weight']).drop_duplicates().reset_index(drop=True)  ## not full; sym

    l_num =861
    d_num=432
    offsets = {'uid' : l_num, 'did' : l_num+d_num}
    print(lm['mid'].max())
    # offsets['mid'] = 1147
    offsets['mid'] = 1730

    #* positive pairs
    ld_pos = ld[ld['rating'] == 1].to_numpy()[:, :2]
    np.save("./preprocessed/LncRNA/dataset2/pos_ratings_offset01", ld_pos)
    # * adjs with offset
    adjs_offset = {}

    # * node types
    node_types = np.zeros((offsets['mid'],), dtype=np.int32)
    node_types[offsets['uid']:offsets['did']] = 1
    node_types[offsets['did']:] = 2
    if not os.path.exists("./preprocessed/LncRNA/dataset2/node_types.npy"):
        np.save("./preprocessed/LncRNA/dataset2/node_types", node_types)
    indices = np.arange(ld_pos.shape[0])
    ## ld
    ld_swap = pd.DataFrame({'lid': ld['lid'], 'did': ld['did'], 'rating': ld['rating']})
    dp_pos_keep = ld_pos[indices]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    u = ld_swap['lid'].tolist()
    v = ld_swap['did'].tolist()
    w = ld_swap['rating'].tolist()
    idx = 0
    for i in range(ld_swap.shape[0]):
        adj_offset[u[idx], v[idx] + offsets['uid']] = w[idx]
        idx = idx + 1
    adjs_offset['1'] = sp.coo_matrix(adj_offset)

    ## dd
    uu_swap = pd.DataFrame({'d1': dd['d1'], 'd2': dd['d2'], 'weight': dd['weight']})
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    u = np.array(uu_swap['d1'].tolist())
    v = np.array(uu_swap['d2'].tolist())
    w = np.array(uu_swap['weight'].tolist())
    idx = 0
    for i in range(uu_swap.shape[0]):
        adj_offset[u[idx] + offsets['uid'], v[idx] + offsets['uid']] = w[idx]
        adj_offset[v[idx] + offsets['uid'], u[idx] + offsets['uid']] = w[idx]
        idx = idx + 1
    adjs_offset['2'] = sp.coo_matrix(adj_offset)

    # lm
    lm_swap = pd.DataFrame({'lid': lm['lid'], 'mid': lm['mid'], 'weight': lm['weight']})
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    u = lm_swap['lid'].tolist()
    v = lm_swap['mid'].tolist()
    w = lm_swap['weight'].tolist()
    idx = 0
    for i in range(lm_swap.shape[0]):
        adj_offset[u[idx], v[idx] + offsets['did']] = w[idx]
        idx = idx + 1
    adjs_offset['3'] = sp.coo_matrix(adj_offset)

    # ll
    uu_swap = pd.DataFrame({'l1': ll['l1'], 'l2': ll['l2'], 'weight': ll['weight']})
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    u = uu_swap['l1'].tolist()
    v = uu_swap['l2'].tolist()
    w = uu_swap['weight'].tolist()
    idx = 0
    for i in range(uu_swap.shape[0]):
        adj_offset[u[idx], v[idx]] = w[idx]
        adj_offset[v[idx], u[idx]] = w[idx]
        idx = idx + 1
    adjs_offset['0'] = sp.coo_matrix(adj_offset)

    # md
    de_npy = md.to_numpy()[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    print(de_npy[:, 0].max())
    print(offsets['did'])
    print(adj_offset.shape)
    adj_offset[de_npy[:, 0] + offsets['did'], de_npy[:, 1] + offsets['uid']] = 1
    adjs_offset['4'] = sp.coo_matrix(adj_offset)

    # mm
    uu_swap = pd.DataFrame({'m1': mm['m1'], 'm2': mm['m2'], 'weight': mm['weight']})
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    u = uu_swap['m1'].tolist()
    v = uu_swap['m2'].tolist()
    w = uu_swap['weight'].tolist()
    idx = 0
    for i in range(uu_swap.shape[0]):
        adj_offset[u[idx] + offsets['did'], v[idx] + offsets['did']] = w[idx]
        adj_offset[v[idx] + offsets['did'], u[idx] + offsets['did']] = w[idx]
        idx = idx + 1
    adjs_offset['5'] = sp.coo_matrix(adj_offset)

    f2 = open("./preprocessed/LncRNA/dataset2/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()
    

    
    #! negative rating
    neg_ratings = ldall[ldall['rating'] == 0].to_numpy()[:, :2]
    # assert(ld_pos.shape[0] + neg_ratings.shape[0] == ldall.shape[0])
    np.save("./preprocessed/LncRNA/dataset2/neg_ratings_offset", neg_ratings)
    # neg_ratings[:, 1] += offsets['uid']
    # SD = np.array(pd.read_excel(os.path.join(prefix, "04-lncRNA-lncRNA.xlsx")))
    # SM = np.array(pd.read_excel(os.path.join(prefix, "07-disease-disease.xlsx")))
    # major = []
    # for i in range(96182):
    #     mm=SD[neg_ratings[i][0], :].tolist()
    #     dd=SM[neg_ratings[i][1], :].tolist()
    #     q = SD[neg_ratings[i][0], :].tolist() + SM[neg_ratings[i][1], :].tolist()
    #     major.append(q)
    # kmeans = KMeans(n_clusters=23, random_state=0).fit(major)
    # center = kmeans.cluster_centers_
    # center_x = []
    # center_y = []
    # for j in range(len(center)):
    #     center_x.append(center[j][0])
    #     center_y.append(center[j][1])
    # labels = kmeans.labels_
    # type1_x = []
    # type1_y = []
    # type2_x = []
    # type2_y = []
    # type3_x = []
    # type3_y = []
    # type4_x = []
    # type4_y = []
    # type5_x = []
    # type5_y = []
    # type6_x = []
    # type6_y = []
    # type7_x = []
    # type7_y = []
    # type8_x = []
    # type8_y = []
    # type9_x = []
    # type9_y = []
    # type10_x = []
    # type10_y = []
    # type11_x = []
    # type11_y = []
    # type12_x = []
    # type12_y = []
    # type13_x = []
    # type13_y = []
    # type14_x = []
    # type14_y = []
    # type15_x = []
    # type15_y = []
    # type16_x = []
    # type16_y = []
    # type17_x = []
    # type17_y = []
    # type18_x = []
    # type18_y = []
    # type19_x = []
    # type19_y = []
    # type20_x = []
    # type20_y = []
    # type21_x = []
    # type21_y = []
    # type22_x = []
    # type22_y = []
    # type23_x = []
    # type23_y = []
    # for i in range(len(labels)):
    #     if labels[i] == 0:
    #         type1_x.append(neg_ratings[i][0])
    #         type1_y.append(neg_ratings[i][1])
    #     if labels[i] == 1:
    #         type2_x.append(neg_ratings[i][0])
    #         type2_y.append(neg_ratings[i][1])
    #     if labels[i] == 2:
    #         type3_x.append(neg_ratings[i][0])
    #         type3_y.append(neg_ratings[i][1])
    #     if labels[i] == 3:
    #         type4_x.append(neg_ratings[i][0])
    #         type4_y.append(neg_ratings[i][1])
    #     if labels[i] == 4:
    #         type5_x.append(neg_ratings[i][0])
    #         type5_y.append(neg_ratings[i][1])
    #     if labels[i] == 5:
    #         type6_x.append(neg_ratings[i][0])
    #         type6_y.append(neg_ratings[i][1])
    #     if labels[i] == 6:
    #         type7_x.append(neg_ratings[i][0])
    #         type7_y.append(neg_ratings[i][1])
    #     if labels[i] == 7:
    #         type8_x.append(neg_ratings[i][0])
    #         type8_y.append(neg_ratings[i][1])
    #     if labels[i] == 8:
    #         type9_x.append(neg_ratings[i][0])
    #         type9_y.append(neg_ratings[i][1])
    #     if labels[i] == 9:
    #         type10_x.append(neg_ratings[i][0])
    #         type10_y.append(neg_ratings[i][1])
    #     if labels[i] == 10:
    #         type11_x.append(neg_ratings[i][0])
    #         type11_y.append(neg_ratings[i][1])
    #     if labels[i] == 11:
    #         type12_x.append(neg_ratings[i][0])
    #         type12_y.append(neg_ratings[i][1])
    #     if labels[i] == 12:
    #         type13_x.append(neg_ratings[i][0])
    #         type13_y.append(neg_ratings[i][1])
    #     if labels[i] == 13:
    #         type14_x.append(neg_ratings[i][0])
    #         type14_y.append(neg_ratings[i][1])
    #     if labels[i] == 14:
    #         type15_x.append(neg_ratings[i][0])
    #         type15_y.append(neg_ratings[i][1])
    #     if labels[i] == 15:
    #         type16_x.append(neg_ratings[i][0])
    #         type16_y.append(neg_ratings[i][1])
    #     if labels[i] == 16:
    #         type17_x.append(neg_ratings[i][0])
    #         type17_y.append(neg_ratings[i][1])
    #     if labels[i] == 17:
    #         type18_x.append(neg_ratings[i][0])
    #         type18_y.append(neg_ratings[i][1])
    #     if labels[i] == 18:
    #         type19_x.append(neg_ratings[i][0])
    #         type19_y.append(neg_ratings[i][1])
    #     if labels[i] == 19:
    #         type20_x.append(neg_ratings[i][0])
    #         type20_y.append(neg_ratings[i][1])
    #     if labels[i] == 20:
    #         type21_x.append(neg_ratings[i][0])
    #         type21_y.append(neg_ratings[i][1])
    #     if labels[i] == 21:
    #         type22_x.append(neg_ratings[i][0])
    #         type22_y.append(neg_ratings[i][1])
    #     if labels[i] == 22:
    #         type23_x.append(neg_ratings[i][0])
    #         type23_y.append(neg_ratings[i][1])
    # type = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # 23簇
    # mtype = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # dataSet = []
    # for k1 in range(len(type1_x)):
    #     type[0].append((type1_x[k1], type1_y[k1]))
    # for k2 in range(len(type2_x)):
    #     type[1].append((type2_x[k2], type2_y[k2]))
    # for k3 in range(len(type3_x)):
    #     type[2].append((type3_x[k3], type3_y[k3]))
    # for k4 in range(len(type4_x)):
    #     type[3].append((type4_x[k4], type4_y[k4]))
    # for k5 in range(len(type5_x)):
    #     type[4].append((type5_x[k5], type5_y[k5]))
    # for k6 in range(len(type6_x)):
    #     type[5].append((type6_x[k6], type6_y[k6]))
    # for k7 in range(len(type7_x)):
    #     type[6].append((type7_x[k7], type7_y[k7]))
    # for k8 in range(len(type8_x)):
    #     type[7].append((type8_x[k8], type8_y[k8]))
    # for k9 in range(len(type9_x)):
    #     type[8].append((type9_x[k9], type9_y[k9]))
    # for k10 in range(len(type10_x)):
    #     type[9].append((type10_x[k10], type10_y[k10]))
    # for k11 in range(len(type11_x)):
    #     type[10].append((type11_x[k11], type11_y[k11]))
    # for k12 in range(len(type12_x)):
    #     type[11].append((type12_x[k12], type12_y[k12]))
    # for k13 in range(len(type13_x)):
    #     type[12].append((type13_x[k13], type13_y[k13]))
    # for k14 in range(len(type14_x)):
    #     type[13].append((type14_x[k14], type14_y[k14]))
    # for k15 in range(len(type15_x)):
    #     type[14].append((type15_x[k15], type15_y[k15]))
    # for k16 in range(len(type16_x)):
    #     type[15].append((type16_x[k16], type16_y[k16]))
    # for k17 in range(len(type17_x)):
    #     type[16].append((type17_x[k17], type17_y[k17]))
    # for k18 in range(len(type18_x)):
    #     type[17].append((type18_x[k18], type18_y[k18]))
    # for k19 in range(len(type19_x)):
    #     type[18].append((type19_x[k19], type19_y[k19]))
    # for k20 in range(len(type20_x)):
    #     type[19].append((type20_x[k20], type20_y[k20]))
    # for k21 in range(len(type21_x)):
    #     type[20].append((type21_x[k21], type21_y[k21]))
    # for k22 in range(len(type22_x)):
    #     type[21].append((type22_x[k22], type22_y[k22]))
    # for k23 in range(len(type23_x)):
    #     type[22].append((type23_x[k23], type23_y[k23]))  # Divide Major into 23 clusters by K-means clustering
    # for k in range(23):
    #     mtype[k] = random.sample(type[k], 240)
    # for m2 in range(240):
    #     for n2 in range(412):
    #         for z2 in range(23):
    #             if (m2, n2) in mtype[z2]:
    #                 dataSet.append((m2, n2))
    #
    # np.save("./preprocessed/LncRNA/neg_ratings_offset02", dataSet)


    # np.random.shuffle(indices)
    # keep, keep = np.array_split(indices, 2)
    # train, val, test = np.array_split(indices, [int(len(indices) * 0.6), int(len(indices) * 0.8)])
    # ld_pos_train = ld_pos[train]
    # ld_pos_val = ld_pos[val]
    # ld_pos_test = ld_pos[test]
    #
    # ld_pos_train[:, 1] += offsets['uid']
    # ld_pos_val[:, 1] += offsets['uid']
    # ld_pos_test[:, 1] += offsets['uid']
    # np.savez("./preprocessed/LncRNA/pos_pairs_offset", train=ld_pos_train, val=ld_pos_val, test=ld_pos_test)
    train, val = np.array_split(indices, [int(len(indices) * 0.8)])
    ld_pos_train = ld_pos[train]
    ld_pos_val = ld_pos[val]

    ld_pos_train[:, 1] += offsets['uid']
    ld_pos_val[:, 1] += offsets['uid']
    ld_pos[:,1] += offsets['uid']
    np.savez("./preprocessed/LncRNA/dataset2/pos_pairs_offsetnew", train=ld_pos_train, val=ld_pos_val,all=ld_pos)

if __name__ == '__main__':
    # prefix = "./data/data"
    prefix = "./data/data/dataset2"
    #sys.argv[]是用来获取命令行输入的参数的(参数和参数之间空格区分),sys.argv[0]表示代码本身文件路径,所以从参数1开始,表示获取的参数了
    preprocess_Drug(prefix)
