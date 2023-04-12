import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv

"""DAG 搜索空间（即超网）"""
# conv1 = GATConv(in_channels=64, out_channels=64, heads=1, add_self_loops=False,bias=True,dropout=0.2).cuda()
# class Op(nn.Module):
#     '''𝑯（3)和𝑯(4）有多个传入链接，它将多个传播路径组合在一起。这不能通过GTN[34]来实现，它只通过矩阵乘法来学习元路径。'''
#     def __init__(self):
#         super(Op, self).__init__()
#     def forward(self, x, adjs, ws, idx):
#         #assert(ws.size(0) == len(adjs))
#         if(adjs[idx].size(0) == 2):
#             return ws[idx] * conv1(x, adjs[idx])
#         elif(adjs[idx].size(0) != 2):
#             return ws[idx] * torch.spmm(adjs[idx], x)

class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, ws, idx):
        return  ws[idx]*torch.spmm(adjs[idx], x)


class Cell(nn.Module):
    '''
    the DAG search space
    '''
    #use_norm:对隐藏层归一化
    def __init__(self, n_step, n_hid_prev, n_hid, cstr, use_norm = True, use_nl = True):
        super(Cell, self).__init__()
        #映射特征维度
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step               #* 中间状态的数量（即K）。
        #LayerNorm实际就是对隐含层做层归一化，即对某一层的所有神经元的输入进行归一化。（每hidden_size个数求平均/方差）
        self.norm = nn.LayerNorm(n_hid, elementwise_affine = False) if use_norm is True else lambda x : x
        self.use_nl = use_nl
        #如果要判断两个类型是否相同推荐使用 isinstance()
        assert(isinstance(cstr, list))
        self.cstr = cstr                   #* 类型约束

        #A ∪ {𝑰}
        self.ops_seq = nn.ModuleList()     #* state (i - 1) -> state i, 1 <= i < K
        for i in range(1, self.n_step):
            self.ops_seq.append(Op())

        #A ∪ {𝑰} ∪ {𝑶}
        self.ops_res = nn.ModuleList()     #* state j -> state i, 0 <= j < i - 1, 2 <= i < K
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(Op())

        #A-
        self.last_seq = Op()               #* state (K - 1) -> state K
        #A- ∪ {𝑰} ∪ {𝑶}
        self.last_res = nn.ModuleList()    #* state i -> state K, 0 <= i < K - 1
        for i in range(self.n_step - 1):
            self.last_res.append(Op())
    

    def forward(self, x, adjs, ws_seq, idxes_seq, ws_res, idxes_res):
        #assert(isinstance(ws_seq, list))
        #assert(len(ws_seq) == 2)
        #1.映射指定维度
        x = self.affine(x)
        #2.图的状态
        states = [x]
        # offset节点类型
        offset = 0
        for i in range(self.n_step - 1):
            # A对指定边缘类型（index_seq）进行消息传递,offset:节点类型
            # idxes_seq:选取哪一个矩阵(A ∪ {𝑰 })
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i], idxes_seq[0][i])  # ! 排除零运算
            # A ∪ {𝑰 } ∪ {𝑶}
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j], idxes_res[0][offset + j]) for j, h in
                       enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        # assert(offset == len(self.ops_res))

        #约束矩阵->消息传播
        adjs_cstr = [adjs[i] for i in self.cstr]#选取哪一个邻接矩阵
        #取最后一个图的状态更新(A-)
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1], idxes_seq[1])
        #添加空矩阵(A- ∪ {𝑰 } ∪ {𝑶})
        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i], idxes_res[1][i]) for i, h in enumerate(states[:-1]))
        # 求范数（1求绝对值，2求平方和开平方，3求平方和开立方根）
        output = self.norm(out_seq + out_res)
        # gelu函数激活
        # if self.use_nl:
        #     output = F.gelu(output)
        return output


class Model(nn.Module):
    #n_steps：我们输入的长度
    #cstr；preprocess中定义大的类型约束
    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model, self).__init__()
        #1.神经网络初始化
        self.cstr = cstr #类型约束
        self.n_adjs = n_adjs #邻接矩阵6个(初始化A)
        self.n_hid = n_hid #隐藏层64维
        self.ws = nn.ModuleList() #权重矩阵(初始化nw)
        #* 节点类型的特定转换
        # in_dims[708,1512]
        assert(isinstance(in_dims, list))
        for i in range(len(in_dims)):
            # 708->64
            # 1512->64
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert(isinstance(n_steps, list))

        #2.药物靶标DAG
        self.metas = nn.ModuleList()
        #一个metas
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, cstr, use_norm = use_norm, use_nl = out_nl))

        #3.架构参数表示
        self.as_seq = []                   #* arch parameters for ops_seq(正态分布)
        self.as_last_seq = []              #* arch parameters for last_seq(正态分布)
        for i in range(len(n_steps)):
            if n_steps[i] > 1:
                #ai = 1e-3*[n_steps[i] - 1,n_adjs - 1](正态分布)
                ai = 1e-3 * torch.randn(n_steps[i] - 1, n_adjs - 1)   #! exclude zero Op
                ai = ai.cuda()
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            #ai_last = 1e-3*[1,1]
            ai_last = 1e-3 * torch.randn(len(cstr))
            ai_last = ai_last.cuda()
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)
        
        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        #as_res根据ks设定
        #ai_last根据n_steps设定
        self.as_res = []                  #* arch parameters for ops_res 
        self.as_last_res = []             #* arch parameters for last_res
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)
                ai = ai.cuda()
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            
            if n_steps[i] > 1:
                #ai_last:生成一个正态分布的数字[1,2]
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1)
                ai_last = ai_last.cuda()
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)
        
        assert(ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2)
        
        #* [optional] combine more than one meta graph? 
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        self.fc3 = nn.Conv1d(128,128,1)

    #将as_seq,as_last_seq,as_res,as_last_res添加到alphas
    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)
        return alphas
    
    def sample(self, eps):
        '''
        对每条链路的一个候选边缘类型进行采样
        '''
        # 采样as_seq最大值的索引
        idxes_seq = []
        # 采样as_res最大值的索引
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]).cuda())
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)).cuda())
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]).cuda())
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]).cuda())
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                # 选出as_seq每行最大值的索引
                temp.append(None if self.as_seq[i] is None else torch.argmax(F.softmax(self.as_seq[i], dim=-1), dim=-1))
                # 选出as_last_seq每行最大值的索引
                temp.append(torch.argmax(F.softmax(self.as_last_seq[i], dim=-1), dim=-1))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                # 选出as_res每行最大值的索引
                temp.append(None if self.as_res[i] is None else torch.argmax(F.softmax(self.as_res[i], dim=-1), dim=-1))
                # 选出as_last_res每行最大值的索引
                temp.append(None if self.as_last_res[i] is None else torch.argmax(F.softmax(self.as_last_res[i], dim=-1), dim=-1))
                idxes_res.append(temp)
        return idxes_seq, idxes_res
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res,semantics,cosins):
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            #每一个节点*ws权重
            hid[node_types == i] = self.ws[i](node_feats[i])
        temps = []; attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1))
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, idxes_seq[i], ws_res, idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)

        #全部消息传递结果拼接
        hids = torch.stack(temps, dim=0).transpose(0, 1)
        #最后一个消息传递结果拼接
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        #消息传递相乘
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        out = out * (1-cosins) + semantics * cosins
        # out = self.gcn(out,)
        # out = torch.cat((out,semantics),dim=1)

        return out


    def parse(self):
        '''
        得出一个由arch参数表示的元图
        '''
        idxes_seq, idxes_res = self.sample(0.)
        msg_seq = []; msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [self.cstr[idxes_seq[i][1].item()]]
            #idxes_seq每个tensor相加
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0].tolist() + map_seq)
            assert(len(msg_seq[i]) == self.metas[i].n_step)

            temp_res = []
            if idxes_res[i][1] is not None:
                for item in idxes_res[i][1].tolist():
                    if item < len(self.cstr):
                        temp_res.append(self.cstr[item])
                    else:
                        assert(item == len(self.cstr))
                        temp_res.append(self.n_adjs - 1)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0].tolist() + temp_res
            assert(len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)
        
        return msg_seq, msg_res