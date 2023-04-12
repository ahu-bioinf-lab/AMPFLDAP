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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc
import pandas as pd
from model import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
from arch import archs
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import random
from sklearn.cluster import KMeans
import xlrd
import xlwt
import xlsxwriter

"""这个脚本import arch.py,从头开始导入和训练发现的架构"""
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.007, help='learning rate')
parser.add_argument('--wd', type=float, default=0.09, help='weight decay')
parser.add_argument('--n_hid', type=int, default=128, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='LncRNA')
parser.add_argument('--dataset2', type=str, default='LncRNA/dataset2')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

logdir = os.path.join("log/eval", args.dataset)
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

    steps_s = [len(meta) for meta in archs[args.dataset]["source"][0]]
    steps_t = [len(meta) for meta in archs[args.dataset]["target"][0]]

    datadir = "preprocessed"
    prefix = os.path.join(datadir, args.dataset2)

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adjs_pt = []
    if '0' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).cuda())
    if '2' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['2'] + sp.eye(adjs_offset['2'].shape[0], dtype=np.float32))).cuda())
    if '5' in adjs_offset:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(adjs_offset['5'] + sp.eye(adjs_offset['5'].shape[0], dtype=np.float32))).cuda())
    for i in range(1, 2):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
    for j in range(3, 5):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(j)] + sp.eye(adjs_offset[str(j)].shape[0], dtype=np.float32))).cuda())
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(adjs_offset[str(j)].T + sp.eye(adjs_offset[str(j)].shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos = np.load(os.path.join(prefix, "pos_pairs_offsetnew.npz"))
    pos_train = pos['train']
    # pos_val = pos['val']
    # pos_test = pos['val']
    pos_all = pos['all']
    #
    neg = np.load(os.path.join(prefix, "neg_pairs_offsetnew03.npz"))
    # neg_train = neg['train']
    # neg_val = neg['val']
    # neg_test = neg['test']
    neg_all = neg['val']


    #* input features
    in_dims = []
    node_feats = []
    # assert(len(in_dims) == len(node_feats))

    # node_feat = pd.read_excel(os.path.join(prefix, "Dataset1_64.xlsx"))
    # node_feat = pd.read_excel(os.path.join(prefix, "node2vec128.xlsx"))
    # node_feat = pd.read_excel(os.path.join(prefix, "struc2vec128.xlsx"))
    # node_feat = pd.read_excel(os.path.join(prefix, "Dataset1_32.xlsx"))
    # node_feat = pd.read_excel(os.path.join(prefix, "Dataset2_128.xlsx"))
    node_feat = pd.read_excel(os.path.join(prefix, "struc2vec_data2_128.xlsx"))
    node_feat=np.array(node_feat)
    # LL = pd.read_excel(os.path.join(prefix, "04-lncRNA-lncRNA.xlsx"))
    # MM = pd.read_excel(os.path.join(prefix, "09-miRNA-miRNA.xlsx"))
    # DD = pd.read_excel(os.path.join(prefix, "07-disease-disease.xlsx"))
    LL = pd.read_excel(os.path.join(prefix, "lncRNA-lncRNA.xlsx"))
    MM = pd.read_excel(os.path.join(prefix, "miRNA-miRNA.xlsx"))
    DD = pd.read_excel(os.path.join(prefix, "disease-disease.xlsx"))
    node_feat = np.array(node_feat)
    LL = np.array(LL)
    MM = np.array(MM)
    DD = np.array(DD)
    # pca降维
    pca_sk = PCA(n_components=128)
    ll = pca_sk.fit_transform(LL)
    mm = pca_sk.fit_transform(MM)
    dd = pca_sk.fit_transform(DD)

    semantics = np.vstack((ll, dd))
    semantics = np.vstack((semantics, mm))
    print(semantics)


    # lncrna_num = 240
    # dis_num = 652
    # mi_num = 1147
    lncrna_num = 861
    dis_num = 1293
    mi_num = 1730
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
            num = node_feat[i].dot(dd[i-861].T)
            denom = np.linalg.norm(node_feat[i]) * np.linalg.norm(dd[i-861])
            cosin = num / denom
            cosins.append(cosin)
        if (dis_num<=i ):
            num = node_feat[i].dot(mm[i-1293].T)
            denom = np.linalg.norm(node_feat[i]) * np.linalg.norm(mm[i-1293])
            cosin = num / denom
            cosins.append(cosin)
    print(cosins)
    cosins =np.array(cosins)
    cosins = (cosins+1)/2
    cosins = cosins.reshape(1730,1)
    cosins = np.tile(cosins,128)
    node_feat = node_feat*cosins+semantics*(1-cosins)
    print(node_feat)
    # node_feats.append(torch.from_numpy(node_feat[0:lncrna_num]).to(torch.float32).cuda())
    # node_feats.append(torch.from_numpy(node_feat[lncrna_num:dis_num]).to(torch.float32).cuda())
    # node_feats.append(torch.from_numpy(node_feat[dis_num:mi_num]).to(torch.float32).cuda())
    cosins = torch.from_numpy(cosins).cuda()
    semantics = torch.from_numpy(semantics).cuda()

    node_feats.append(torch.from_numpy(node_feat[0:lncrna_num]).to(torch.float32).cuda())
    node_feats.append(torch.from_numpy(node_feat[lncrna_num:dis_num]).to(torch.float32).cuda())
    node_feats.append(torch.from_numpy(node_feat[dis_num:mi_num]).to(torch.float32).cuda())
    node_feats = []

    for k in range(num_node_types):
        in_dims.append((node_types == k).sum().item())
        i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
        v = torch.ones(in_dims[-1])
        node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())

    for k in range(num_node_types):
        # in_dims.append((node_types == k).sum().item())
        in_dims.append(128)



    model_s = Model(in_dims, args.n_hid, steps_s, dropout = args.dropout).cuda()
    model_t = Model(in_dims, args.n_hid, steps_t, dropout = args.dropout).cuda()

    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )
    #开始训练
    auc1 = []
    aupr1 = []
    acc1 = []
    precision1 = []
    recall1 = []
    F11 = []
    auprallbest1 =[]
    auc_valallbest1 =[]
    auc2 = []
    aupr2 = []
    acc2 = []
    precision2 = []
    recall2 = []
    F12 = []
    auc_valallmean2=[]
    auprallbestmean2=[]
    tprs = []
    aucs = []

    K=5
    turns = 10
    for turns in range(turns):
        print('*' * 25,'第', turns + 1, '次五折交叉验证','*' * 25)
        pos_all =list(pos_all)
        neg_all =list(neg_all)
        for i in range(K):
            print('*' * 25, i + 1, '*' * 25)
            data_train_pos,data_valid_pos,data_train_neg,data_valid_neg,data_neg = get_k_fold_data(K, i, pos_all,neg_all)
            data_valid_neg = np.array(data_valid_neg)
            data_valid_pos = np.array(data_valid_pos)
            data_train_pos = np.array(data_train_pos)
            data_train_neg = np.array(data_train_neg)
            data_neg = np.array(data_neg)
            best_test = None
            final = None
            anchor = None
            predict_best = None
            auc_best = None
            aupr_best = None
            acc_best = None
            precision_best = None
            recall_best = None
            f1_best = None
            y_true_val_best =None
            y_val_best =None
            auc_valallbest=None
            auprallbest =None
            best_test1 = None
            y_true_val_best1=None
            y_val_best1=None
            for epoch in range(args.epochs):
                train_loss = train(node_feats, node_types, adjs_pt, data_train_pos, data_train_neg, model_s, model_t, optimizer,cosins,semantics)
                val_loss, auc_val,auc_valall,auprall, auc_test, aupr, acc, precision, recall, f1, predict,y_true_val,y_val = infer(node_feats, node_types, adjs_pt, data_valid_pos, data_valid_neg, data_train_pos, data_train_neg, model_s, model_t,cosins,semantics,pos_all,data_neg)
                print("第{}轮; 训练损失：{}; 验证损失：{}; 训练正确率：{}; 测试正确率：{};".format(epoch + 1, train_loss, val_loss, auc_val, auc_test))
                auc_val = roc_auc_score(y_true_val, y_val)
                # logging.info("Epoch {}; Train err {}; Val err {}; Test auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
                if best_test is None or auc_test > best_test:
                    best_test = auc_test
                    y_true_val_best = y_true_val
                    y_val_best = y_val
                if aupr_best is None or aupr > aupr_best:
                    aupr_best = aupr
                if acc_best is None or acc > acc_best:
                    acc_best = acc
                if precision_best is None or precision > precision_best:
                    precision_best = precision
                if recall_best is None or recall > recall_best:
                    recall_best = recall
                if f1_best is None or f1 > f1_best:
                    f1_best = f1
                    # precision_best = precision
                    # recall_best = recall
                    # f1_best = f1
                    predict_best = predict
                if auc_valallbest is None or auc_valall > auc_valallbest:
                    auc_valallbest = auc_valall
                if auprallbest is None or auprall > auprallbest:
                    auprallbest = auprall

            print("AUC：{:.6f}".format(best_test))
            print("AUPR：{:.6f}".format(aupr_best))
            print("ACC：{:.6f}".format(acc_best))
            print("PRECISION：{:.6f}".format(precision_best))
            print("RECALL：{:.6f}".format(recall_best))
            print("F1：{:.6f}".format(f1_best))
            print("最好结果：{}".format(predict_best))
            print("非平衡AUC：{}".format(auc_valallbest))
            print("非平衡AUPR：{}".format(auprallbest))
            auc1.append(best_test)
            aupr1.append(aupr_best)
            acc1.append(acc_best)
            precision1.append(precision_best)
            recall1.append(recall_best)
            F11.append(f1_best)
            auc_valallbest1.append(auc_valallbest)
            auprallbest1.append(auprallbest)

            if best_test1 is None or best_test1 < best_test:
                best_test1 = best_test
                y_true_val_best1=y_true_val_best
                y_val_best1=y_val_best

            #计算roc值
        mean_fpr = np.linspace(0, 1, 1000)
        fpr, tpr, rocth = roc_curve(y_true_val_best1, y_val_best1)
        auc_val = roc_auc_score(y_true_val_best1, y_val_best1)
        auroc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auroc)
        auc_mean =np.mean(np.array(auc1))
        aupr_mean =np.mean(np.array(aupr1))
        acc_mean = np.mean(np.array(acc1))
        precision_mean =np.mean(np.array(precision1))
        recall_mean =np.mean(np.array(recall1))
        F1_mean =np.mean(np.array(F11))
        auc_valallbestmean = np.mean(auc_valallbest1)
        auprallbestmean = np.mean(auprallbest1)
        print("AUC_mean：{:.6f}".format(auc_mean))
        print("AUPR_mean：{:.6f}".format(aupr_mean))
        print("ACC_mean：{:.6f}".format(acc_mean))
        print("PRECISION_mean：{:.6f}".format(precision_mean))
        print("RECALL_mean：{:.6f}".format(recall_mean))
        print("F1_mean：{:.6f}".format(F1_mean))
        print("auc_valallmean：{:.6f}".format(auc_valallbestmean))
        print("auprallmean：{:.6f}".format(auprallbestmean))
        auc2.append(auc_mean)
        aupr2.append(aupr_mean)
        acc2.append(acc_mean)
        precision2.append(precision_mean)
        recall2.append(recall_mean)
        F12.append(F1_mean)
        auc_valallmean2.append(auc_valallbestmean)
        auprallbestmean2.append(auprallbestmean)
        pos_all =random.sample(list(pos_all),4518)
        neg_all = random.sample(list(neg_all), 367434)
        # pos_all =random.sample(list(pos_all),2697)
        # neg_all = random.sample(list(neg_all), 2697)
    # show_auc(tprs, aucs)
    tprs=np.array(tprs)
    tprs=pd.DataFrame(tprs)
    tprs.to_excel("D:\\Python_work\\DIFFMG\\LncRNA-diseases\\LncRNA-diseases\\preprocessed\\LncRNA\\dataset2\\tprs\\tprs.xlsx")
    aucs=np.array(aucs)
    aucs=pd.DataFrame(aucs)
    aucs.to_excel("D:\\Python_work\\DIFFMG\\LncRNA-diseases\\LncRNA-diseases\\preprocessed\\LncRNA\\dataset2\\tprs\\aucs.xlsx")
    auc_mean10 = np.mean(np.array(auc2))
    aupr_mean10 = np.mean(np.array(aupr2))
    acc_mean10 = np.mean(np.array(acc2))
    precision_mean10 = np.mean(np.array(precision2))
    recall_mean10 = np.mean(np.array(recall2))
    F1_mean10 = np.mean(np.array(F12))
    auc_valallmean10 = np.mean(np.array(auc_valallmean2))
    auprallbestmean10 = np.mean(np.array(auprallbestmean2))
    print("AUC_mean10：{:.6f}".format(auc_mean10))
    print("AUPR_mean10：{:.6f}".format(aupr_mean10))
    print("ACC_mean10：{:.6f}".format(acc_mean10))
    print("PRECISION_mean10：{:.6f}".format(precision_mean10))
    print("RECALL_mean10：{:.6f}".format(recall_mean10))
    print("F1_mean10：{:.6f}".format(F1_mean10))
    print("auc_valallmean10：{:.6f}".format(auc_valallmean10))
    print("auprallbestmean10：{:.6f}".format(auprallbestmean10))
    show_auc(tprs, aucs)


def show_auc(tprs, aucs):
    mean_fpr = np.linspace(0, 1,1000)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Five fold Cross-Validation')
    plt.legend(loc="lower right")
    plt.show()

def get_k_fold_data(K, i, pos_all,neg_all):
    assert K>1
    # data_pos=pos_all[0:2697]
    data_pos = pos_all[0:4518]
    data_neg=neg_all
    # data_valid_pos =[]
    # data_train_pos =[]
    # kf = KFold(n_splits=5, shuffle=True)
    # for train, test in kf.split(pos_all):
    #     arr = np.array(pos_all)
    #     data_train_pos.append(arr[train].tolist())
    #     data_valid_pos.append(arr[test].tolist())
    # start=int(i*2697//K)
    # end=int((i+1)*2697//K)
    start=int(i*4518//K)
    end=int((i+1)*4518//K)

    data_train, data_valid=None, None
    data_valid_pos, data_valid_neg=None, None
    data_train_pos, data_train_neg=None, None

    data_valid_pos=data_pos[start:end]
    # data_train_pos1=data_pos[0:start]
    # data_train_pos2=data_pos[end:2697]
    # data_train_pos=np.vstack((data_train_pos1,data_train_pos2))
    # data_train_pos=data_pos[0:start]+data_pos[end:2697]
    data_train_pos=data_pos[0:start]+data_pos[end:4518]
    data_valid_neg=data_neg[start:end]
    # data_train_neg=data_neg[0:start]+data_neg[end:2697]
    data_train_neg=data_neg[0:start]+data_neg[end:4518]
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
    # data_neg=np.array(dataSet[0:2697])
    # data_valid_neg=data_neg[start:end]
    # data_train_neg1=data_neg[0:start]
    # data_train_neg2=data_neg[end:2697]
    # data_train_neg = np.vstack((data_train_neg1,data_train_neg2))

    return data_train_pos,data_valid_pos,data_train_neg,data_valid_neg,data_neg


def train(node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer,cosins,semantics):

    model_s.train()
    model_t.train()
    optimizer.zero_grad()
    out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1],cosins,semantics)
    out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1],cosins,semantics)
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def k_svd(X, k):
    # 奇异值分解
    U, Sigma, VT = np.linalg.svd(X)  # 已经自动排序了
    # 数据集矩阵 奇异值分解  返回的Sigma 仅为对角线上的值

    indexVec = np.argsort(-Sigma)  # 对奇异值从大到小排序，返回索引

    # 根据求得的分解，取出前k大的奇异值对应的U,Sigma,V
    K_index = indexVec[:k]  # 取出前k最大的特征值的索引

    U = U[:, K_index]  # 从U取出前k大的奇异值的对应(按列取)
    S = [[0.0 for i in range(k)] for i in range(k)]
    Sigma = Sigma[K_index]  # 从Sigma取出前k大的奇异值(按列取)
    for i in range(k):
        S[i][i] = Sigma[i]  # 奇异值list形成矩阵
    VT = VT[K_index, :]  # 从VT取出前k大的奇异值的对应(按行取)
    X = np.dot(U, S)
    X = np.dot(X, VT)
    return X
def infer(node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t,cosins,semantics,pos_all,data_neg):

    pos_all = np.array(pos_all)
    data_neg =np.array(data_neg)
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs[args.dataset]["source"][0], archs[args.dataset]["source"][1],cosins,semantics)
        out_t = model_t(node_feats, node_types, adjs, archs[args.dataset]["target"][0], archs[args.dataset]["target"][1],cosins,semantics)
    
    #* validation performance
    #正对集合
    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    #负对集合
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)
    pos_val_prodall = torch.mul(out_s[pos_all[:, 0]], out_t[pos_all[:, 1]]).sum(dim=-1)
    neg_val_prodall = torch.mul(out_s[data_neg[:, 0]], out_t[data_neg[:, 1]]).sum(dim=-1)
    # loss = - torch.mean(F.logsigmoid(pos_test_prod) + F.logsigmoid(- neg_test_prod))
    loss = - torch.mean(F.logsigmoid(pos_val_prodall) ) -torch.mean(F.logsigmoid(- neg_val_prodall))

    #创建y_true真实数组
    y_true_test = np.zeros((pos_test.shape[0] + pos_test.shape[0]), dtype=np.int64)
    y_true_all= np.zeros((pos_all.shape[0] + data_neg.shape[0]), dtype=np.int64)
    #y_true pos为1 neg为0
    y_true_test[:pos_test.shape[0]] = 1
    y_true_all[:4518] = 1     # dataset2
    #y_pred将正对集和负对集转换成numpy进行拼接
    y_pred_all = np.concatenate((torch.sigmoid(pos_val_prodall).cpu().numpy(), torch.sigmoid(neg_val_prodall).cpu().numpy()))
    y_pred_test = np.concatenate((torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    #计算准确率
    auc_val = roc_auc_score(y_true_test, y_pred_test)


    #* test performance
    #生成正对集
    pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
    #生成负对集
    neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
    # 生成所有负样本对集
    # neg_all_prod = torch.mul(out_s[neg_val_prodall[:, 0]], out_t[neg_val_prodall[:, 1]]).sum(dim=-1)

    #创建y_true真实数组
    y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.int64)
    #y_true pos为1 neg为0
    y_true_val[:pos_val.shape[0]] = 1
    #y_pred将正对集和负对集转换成numpy进行拼接
    y_pred_val = np.concatenate((torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    y_test = []
    for p in y_pred_test:
        if p>=0.8:
            y_test.append(1)
        else:
            y_test.append(0)
    y_val = []
    for p in y_pred_val:
        if p >= 0.8:
            y_val.append(1)
        else:
            y_val.append(0)
    # auc_test = roc_auc_score(y_true_test, y_pred_test)
    # aupr = average_precision_score(y_true_test, y_pred_test)
    #
    # acc = accuracy_score(y_true_test,y_test)
    # precision = precision_score(y_true_test,y_test)
    # recall = recall_score(y_true_test,y_test)
    # f1 = f1_score(y_true_test,y_test)
    auc_test = roc_auc_score(y_true_val, y_pred_val)
    aupr = average_precision_score(y_true_val, y_pred_val)
    # 非平衡数据集
    auc_valall = roc_auc_score(y_true_all, y_pred_all)
    auprall = average_precision_score(y_true_all, y_pred_all)

    acc = accuracy_score(y_true_val, y_val)
    precision = precision_score(y_true_val, y_val)
    recall = recall_score(y_true_val, y_val)
    f1 = f1_score(y_true_val, y_val)

    # predict
    # p_neg = torch.sigmoid(neg_test_prod)
    # y_predict = []
    # i = 0
    # neg_test1 = neg_test.tolist()
    # for neg in neg_test1:
    #     neg[1] = neg[1]- 240
    #     if p_neg[i]>0.80:
    #         y_predict.append(neg)
    #     i = i+1
    # 预测
    # p_neg = torch.sigmoid(neg_val_prodall)
    # y_predict = []
    # i = 0
    # # neg_test1 = neg_val.tolist()
    # neg_test1 = data_neg.tolist()
    # for neg in neg_test1:
    #     neg[1] = neg[1]- 240
    #     if p_neg[i]>0.99:
    #         y_predict.append(neg)
    #     i = i+1
    p_neg = torch.sigmoid(neg_val_prodall)
    y_predict = []
    i = 0
    k = 0
    # table2 = xlsxwriter.Workbook("predict.xlsx")
    # table1 = table2.add_worksheet("predict")
    # new_table = []
    # neg_test1 = data_neg.tolist()
    # for neg in neg_test1:
    #     neg[1] = neg[1]- 240
    #     if p_neg[i]>0.99:
    #         y_predict.append(neg)
    #         table1.write(k,0,neg[0])
    #         table1.write(k, 1, neg[1])
    #         table1.write(k,2,p_neg[i])
    #         k =k+1
    #     i = i+1
    # table2.close()

    return loss.item(), auc_val,auc_valall,auprall, auc_test, aupr, acc, precision, recall, f1, y_predict,y_true_val,y_pred_val

if __name__ == '__main__':
    main()