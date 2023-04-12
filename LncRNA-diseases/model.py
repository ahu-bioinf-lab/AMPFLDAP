import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

"""从搜索空间派生的任意架构的包装"""

class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()
    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)

class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm = True, use_nl = True):
        super(Cell, self).__init__()
        
        self.affine = nn.Linear(n_hid_prev, n_hid) 
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid) if use_norm is True else lambda x : x
        self.use_nl = use_nl          
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(Op())

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
    
    def forward(self, x, adjs, idxes_seq, idxes_res):
        
        x = self.affine(x)
        states = [x]
        offset = 0
        k=64
        c=0
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i]) #! exclude zero Op

            resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        output = self.norm(states[-1])
        # if self.use_nl:
        #     output = F.gelu(output)
        return output





class Model(nn.Module):

    def __init__(self, in_dims, n_hid, n_steps, dropout = None, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model, self).__init__()
        self.n_hid = n_hid
        self.ws = nn.ModuleList()
        assert(isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert(isinstance(n_steps, list))
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, use_norm = use_norm, use_nl = out_nl))
        # self.weight_lnc = nn.Parameter(torch.Tensor(1, 1147, 1147))
        # self.weight_lnc = init.kaiming_uniform_(self.weight_lnc)
        #* [Optional] Combine more than one meta graph?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        self.fc3 = nn.Linear(128,128)

        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x : x

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

    def crf_layer(self, hidden, hidden_new):
        #
        alpha = 50
        beta = 20

        hidden_extend = hidden.float().unsqueeze(0)

        #   attention
        conv1 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        hidden_extend = hidden_extend.permute(0, 2, 1).cpu()
        seq_fts = conv1(hidden_extend).cpu()

        conv1 = torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        f_1 = conv1(seq_fts).cuda()
        conv1 = torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
        f_2 = conv1(seq_fts).cuda()
        logits = f_1 + f_2.permute(0, 2, 1)

        m = torch.nn.LeakyReLU(0.1)

        coefs = torch.nn.functional.softmax(m(logits) + self.weight_lnc)
        coefs = coefs[0]

        # fenzi
        coefs = coefs.float()
        hidden_new = hidden_new.float().cuda()
        res = torch.mm(coefs, hidden_new)
        hidden_neighbor = torch.mul(res, beta)
        hidden = hidden.float()
        hidden_self = torch.mul(hidden, alpha)
        hidden_crf = hidden_neighbor + hidden_self
        # fenmu
        unit_mat = torch.ones(hidden.shape[0], hidden.shape[1]).float().cuda()
        res = torch.mm(coefs, unit_mat)
        coff_sum = torch.mul(res, beta)
        const = coff_sum + torch.mul(unit_mat, alpha)
        #
        hidden_crf = torch.div(hidden_crf, const)

        return hidden_crf

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res,cosins,semantics):
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)
        # hid =self.crf_layer(hid, hid)
        temps = []; attns = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)
        
        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        # out = out * (1 - cosins) + semantics * cosins
        # out = torch.cat((out,semantics),1)
        # out = torch.sigmoid(out)
        return out