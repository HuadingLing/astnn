import torch.nn as nn
import torch.nn.functional as F
import torch
import random


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, device, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.batch_size = batch_size
        self.node_list = []
        self.batch_node = None
        #self.max_index = vocab_size
        self.device = device
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = torch.zeros(size, self.embedding_dim, device=self.device)

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] != -1:
                index.append(i)  # 最后 index 不就是 [0, 1, ..., size-1] 吗？
                current_node.append(node[i][0])
                temp = node[i][1:]  # node[i][0] 的所有子树
                c_num = len(temp)  # node[i][0] 的孩子个数
                for j in range(c_num):
                    # temp[j][0] 是 node[i][0] 的第j个孩子节点（子树根节点）
                    if temp[j][0] != -1:  # -1是从哪来的？
                        # 应该是从BatchProgramCC过来的，表明 node[i][0] 没有第j个孩子节点（子树根节点）
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)  # ？？
                            children[j].append(temp[j])  # 分组，第j棵子树进入小组children[j]
                    #else: break???
            # else:
            #     batch_index[i] = -1
        # 公式（2）第一项
        batch_current = self.W_c(batch_current.index_copy(0, torch.LongTensor(index, device=self.device),
                                                          self.embedding(torch.LongTensor(current_node, device=self.device))))

        for c in range(len(children)):
            zeros = torch.zeros(size, self.encode_dim, device=self.device)
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, torch.LongTensor(children_index[c], device=self.device), tree)
        # batch_index = [i for i in batch_index if i != -1]
        b_in = torch.LongTensor(batch_index, device=self.device)
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = torch.zeros(self.batch_size, self.encode_dim, device=self.device)  # 用来存储计算结果？应该是用来存储树根的计算结果
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, device, pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size  # 论文中的 |V| (=max_tokens+1)
        self.embedding_dim = embedding_dim  # 论文中的 |d| (=128)
        self.encode_dim = encode_dim  # 论文中的 k (=128)
        self.num_layers = 1
        self.hidden_dim = hidden_dim  # 论文中的 m (=100)
        self.label_size = label_size  # =1
        self.device = device
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.device, pretrained_weight)
        #self.root2label = nn.Linear(self.encode_dim, self.label_size)  # ？？？

        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if isinstance(self.bigru, nn.LSTM):
            h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
            c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
            return h0, c0
        return torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim, device=self.device)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)

    def get_zeros(self, num):
        return torch.zeros(num, self.encode_dim, device=self.device)

    def encode(self, x):
        lens = [len(item) for item in x]  # 求一个batch中每个样本的长度, 这里的len(item)就是对应后面输入GRU的seq_len?
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])  # 将每个样本进行拆分然后全部连接起来，后面的重组要用到 lens

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)  # bigru's input of shape (batch, seq_len, encode_dim)

        gru_out, hidden = self.bigru(encodes, self.hidden)  # gru_out.shape = (batch_size, seq_len, 2*hidden_dim)
        gru_out = torch.transpose(gru_out, 1, 2)  # gru_out.shape = (batch_size, 2*hidden_dim, seq_len)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)  # gru_out.shape = (batch_size, 2*hidden_dim)
        # gru_out = gru_out[:,-1]

        return gru_out

    def forward(self, x):
        features = self.encode(x)
        outputs = self.hidden2label(features)
        return features, outputs