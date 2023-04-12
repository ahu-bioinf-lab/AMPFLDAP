import os
import sys
import time
import numpy as np
import pickle
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
import time
import pandas as pd
from model_search import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor 
from preprocess import cstr_source, cstr_target
from sklearn.decomposition import PCA
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans


"""执行搜索的脚本"""
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.009, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=128, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--steps_s', type=int, default=3,nargs='+', help='number of intermediate states in the meta graph for source node type')
parser.add_argument('--steps_t', type=int, default=3,nargs='+', help='number of intermediate states in the meta graph for target node type')
parser.add_argument('--dataset', type=str, default='LncRNA')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0., help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.8, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_s" + str(args.steps_s) + "_t" + str(args.steps_t) + "_epoch" + str(args.epochs) + \
         "_cuda" + str(args.gpu) + "_eps" + str(args.eps) + "_d" + str(args.decay)

logdir = os.path.join("log/search", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "preprocessed"
    prefix = os.path.join(datadir, args.dataset)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    "两个个D(-1/2)AD(-1/2)和两个D(-1)A"
    # 针对相似矩阵
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).cuda())
    if '2' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['2'] + sp.eye(adjs_offset['2'].shape[0], dtype=np.float32))).cuda())
    if '5' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['5'] + sp.eye(adjs_offset['5'].shape[0], dtype=np.float32))).cuda())

    # 针对两种节点间邻接矩阵
    for i in range(1,2):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    for j in range(3,5):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(j)] + sp.eye(adjs_offset[str(j)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(j)].T + sp.eye(adjs_offset[str(j)].shape[0], dtype=np.float32))).cuda())
    # 添加一个对角矩阵
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    # 添加空矩阵
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    # pos = np.load(os.path.join(prefix, "pos_pairs_offset.npz"))
    # pos_train = pos['train']
    # pos_val = pos['val']
    # pos_test = pos['test']
    #     
    # neg = np.load(os.path.join(prefix, "neg_pairs_offset.npz"))
    # neg_train = neg['train']
    # neg_val = neg['val']
    # neg_test = neg['test']

    pos = np.load(os.path.join(prefix, "pos_pairs_offsetnew.npz"))
    pos_train = pos['train']
    pos_val = pos['val']
    pos_all =pos['all']


    neg = np.load(os.path.join(prefix, "neg_pairs_offsetnew02.npz"))
    neg_train = neg['train']
    neg_val = neg['val']

    neg = np.load(os.path.join(prefix, "neg_pairs_offsetnew03.npz"))
    neg_train = neg['train']
    neg_val = neg['test']
    neg_all = neg['val']

    #* one-hot IDs as input features
    in_dims = []
    node_feats = []
    # for k in range(num_node_types):
    #     in_dims.append((node_types == k).sum().item())
    #     i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
    #     v = torch.ones(in_dims[-1])
    #     node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
    # assert(len(in_dims) == len(node_feats))
    node_feat = pd.read_excel(os.path.join(prefix, "node2vec128.xlsx"))
    LL = pd.read_excel(os.path.join(prefix, "04-lncRNA-lncRNA.xlsx"))
    MM = pd.read_excel(os.path.join(prefix, "09-miRNA-miRNA.xlsx"))
    DD = pd.read_excel(os.path.join(prefix, "07-disease-disease.xlsx"))
    node_feat=np.array(node_feat)
    LL = np.array(LL)
    MM = np.array(MM)
    DD=np.array(DD)
    # pca降维
    pca_sk = PCA(n_components=128)
    ll = pca_sk.fit_transform(LL)
    mm = pca_sk.fit_transform(MM)
    dd = pca_sk.fit_transform(DD)
    semantics = np.vstack((ll,dd))
    semantics = np.vstack((semantics,mm))
    print(semantics)


    lncrna_num = 240
    dis_num = 652
    mi_num = 1147
    node_feats.append(torch.from_numpy(node_feat[0:lncrna_num]).to(torch.float32).cuda())
    node_feats.append(torch.from_numpy(node_feat[lncrna_num:dis_num]).to(torch.float32).cuda())
    node_feats.append(torch.from_numpy(node_feat[dis_num:mi_num]).to(torch.float32).cuda())
    # ll = torch.from_numpy(ll)
    # mm = torch.from_numpy(mm)
    # dd = torch.from_numpy(dd)
    # node_feat = torch.from_numpy(node_feat)
    cosins=[]
    # 计算余弦相似度
    for i in range(mi_num):
        if(i<lncrna_num):
            num = node_feat[i].dot(ll[i].T)
            denom = np.linalg.norm(node_feat[i]) * np.linalg.norm(ll[i])
            cosin = num / denom
            # cosin = F.cosine_similarity(node_feat[i],ll[i],dim=0)
            cosins.append(cosin)
        if(lncrna_num<=i<dis_num):
            num = node_feat[i].dot(dd[i-240].T)
            denom = np.linalg.norm(node_feat[i]) * np.linalg.norm(dd[i-240])
            cosin = num / denom
            cosins.append(cosin)
        if (dis_num<=i ):
            num = node_feat[i].dot(mm[i-652].T)
            denom = np.linalg.norm(node_feat[i]) * np.linalg.norm(mm[i-652])
            cosin = num / denom
            cosins.append(cosin)
    cosins =np.array(cosins)
    cosins = (cosins+1)/2
    cosins = cosins.reshape(1147,1)
    cosins = np.tile(cosins,128)
    node_feat = node_feat*cosins+semantics*(1-cosins)
    # node_feats.append(torch.from_numpy(node_feat[0:lncrna_num]).to(torch.float32).cuda())
    # node_feats.append(torch.from_numpy(node_feat[lncrna_num:dis_num]).to(torch.float32).cuda())
    # node_feats.append(torch.from_numpy(node_feat[dis_num:mi_num]).to(torch.float32).cuda())
    cosins = torch.from_numpy(cosins).cuda()
    semantics = torch.from_numpy(semantics).cuda()




    for k in range(num_node_types):
        # in_dims.append((node_types == k).sum().item())
        in_dims.append(128)



    #模型初始化
    model_s = Model(in_dims, args.n_hid, len(adjs_pt), args.steps_s, cstr_source['LncRNA']).cuda()
    model_t = Model(in_dims, args.n_hid, len(adjs_pt), args.steps_t, cstr_target['diseases']).cuda()

    optimizer_w = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model_s.alphas() + model_t.alphas(),
        lr=args.alr
    )

    eps = args.eps
    start_t = time.time()
    #开始训练
    K=5
    for i in range(K):
        print('*' * 25, i + 1, '*' * 25)
        data_train_pos,data_valid_pos,data_train_neg,data_valid_neg = get_k_fold_data(K, i, pos_all,neg_all)
        for epoch in range(args.epochs):
            train_error, val_error = train(node_feats, node_types, adjs_pt, data_train_pos, data_train_neg, data_valid_pos, data_valid_neg,
                                           model_s, model_t, optimizer_w, optimizer_a, eps, semantics, cosins,)
            # train_error, val_error = train(node_feats, node_types, adjs_pt, pos_train, neg_train, pos_val, neg_val,
            #                                model_s, model_t, optimizer_w, optimizer_a, eps, semantics, cosins, neg_test)
            s = model_s.parse()
            t = model_t.parse()
            print("Epoch {}; Train err {}; Val err {}; Source arch {}; Target arch {}".format(epoch + 1, train_error,
                                                                                              val_error, s, t))
            eps = eps * args.decay
            best_val= None
            Source_arch=[]
            Target_arch=[]
            if best_val is None or val_error < best_val:
                best_val = val_error
                Source_arch = s
                Target_arch = t

        print("best_val：{}".format(best_val))
        print("Source_arch：{}".format(Source_arch))
        print("Target_arch：{}".format(Target_arch))


        # end_t = time.time()

    #     for epoch in range(args.epochs):
    #         train_error, val_error = train(node_feats, node_types, adjs_pt, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps,semantics,cosins,neg_test)
    #         s = model_s.parse()
    #         t = model_t.parse()
    #         print("Epoch {}; Train err {}; Val err {}; Source arch {}; Target arch {}".format(epoch + 1, train_error, val_error, s, t))
    #         eps = eps * args.decay
    # end_t = time.time()
    # print("搜索时间 (in minutes): {}".format((end_t - start_t) / 60))
def get_k_fold_data(K, i, pos_all,neg_all):
    assert K>1
    data_pos=pos_all[0:2697]
    data_neg=neg_all

    start=int(i*2697//K)
    end=int((i+1)*2697//K)

    data_train, data_valid=None, None
    data_valid_pos, data_valid_neg=None, None
    data_train_pos, data_train_neg=None, None

    data_valid_pos=data_pos[start:end]
    data_train_pos1=data_pos[0:start]
    data_train_pos2=data_pos[end:2697]
    data_train_pos=np.vstack((data_train_pos1,data_train_pos2))
    # data_train_pos=data_pos[0:start]+data_pos[end:2697]
    # data_valid_neg=data_neg[start:end]
    # data_train_neg=data_neg[0:start]+data_neg[end:2697]
    # data_train=data_train_pos+data_train_neg
    # data_valid=data_valid_pos+data_valid_neg
    #! negative rating
    # neg_ratings=neg_all
    # neg_ratings[:,1] -= 240
    # SD = np.array(pd.read_excel(os.path.join("./data/data", "04-lncRNA-lncRNA.xlsx")))
    # SM = np.array(pd.read_excel(os.path.join("./data/data", "07-disease-disease.xlsx")))
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
    #     mtype[k] = random.sample(type[k], 118)
    # for m2 in range(240):
    #     for n2 in range(412):
    #         for z2 in range(23):
    #             if (m2, n2) in mtype[z2]:
    #                 dataSet.append((m2, n2))
    # data_neg=dataSet[0:2697]
    data_valid_neg=data_neg[start:end]
    data_train_neg1=data_neg[0:start]
    data_train_neg2=data_neg[end:2697]
    data_train_neg = np.vstack((data_train_neg1,data_train_neg2))

    return data_train_pos,data_valid_pos,data_train_neg,data_valid_neg



def train(node_feats, node_types, adjs, pos_train, neg_train, pos_val, neg_val, model_s, model_t, optimizer_w, optimizer_a, eps,semantics,cosins):

    #1.对每条链路的一个候选边缘类型进行采样（50%概率）
    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    #2.对w和a交替更新训练
    optimizer_w.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s,semantics,cosins)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t,semantics,cosins)
    loss_w = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, idxes_seq_s, idxes_res_s,semantics,cosins)
    out_t = model_t(node_feats, node_types, adjs, idxes_seq_t, idxes_res_t,semantics,cosins)
    loss_a = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)))
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()

if __name__ == '__main__':
    main()