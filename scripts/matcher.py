import torch as T
from torch.nn import *
from torch import nn
import torch.nn.functional as F
import math
from pytorch_pretrained_bert.modeling import BertLayerNorm


#
# def init_ff_layer(layer, f1=None):
#     weight_size = layer.weight.data.size()[0]
#     if not f1:
#         f1 = 1 / math.sqrt(weight_size)
#     nn.init.uniform_(layer.weight.data, -f1, f1)
#     nn.init.uniform_(layer.bias.data, -f1, f1)

def init_bert_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class EuclideanMatcher(Module):
    def __init__(self):
        super(EuclideanMatcher, self).__init__()
        self.dist = T.cdist

    def forward(self, a, b):
        return self.dist(a, b)

class CosMatcher(Module):
    def __init__(self):
        super(CosMatcher, self).__init__()
        self.cos = CosineSimilarity(dim=1)

    def forward(self, a, b):
        return (self.cos(a,b)+1)/2

class EuclideanLoss(Module):
    def __init__(self, eps=1e-6):
        super(EuclideanLoss, self).__init__()
        self.eps = eps

    def forward(self, queries, centers, ind):
        
        dists = T.cdist(queries, centers)+self.eps
        lossp = dists[:, ind].sum(0)
        tmp1 = dists[:, :ind]
        tmp2 = dists[:, ind + 1:]
        tmp = T.cat([tmp1, tmp2], dim=1)
        exp_sum = T.exp(-tmp).sum(1)
        lossn = T.log(exp_sum).sum(0)
        loss = (lossp + lossn) / queries.size(0)
        return loss

class DotMatcher(Module):
    def forward(self, a, b):
        return F.dropout(a.view(a.size(0),1,-1).bmm(b.view(b.size(0),-1,1)), p=0.2)

class SimpleClassifier(Module):
    def __init__(self, n_class, dim=768, inner_dim=300, dropout=0.3):
        super(SimpleClassifier, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(0.3)
        self.fc_hidden = Linear(dim, inner_dim)
        # init_ff_layer(self.fc_hidden)
        self.fc = Linear(dim, n_class)
        # init_ff_layer(self.fc)
        

    def forward(self, x):
        #x = F.relu(self.fc_hidden(x))
        return self.dropout(self.fc(x))
        return self.fc(x)


class SimpleRegressor(Module):
    def __init__(self, dim=768, dropout=0.3):
        super(SimpleRegressor, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.fc = Linear(dim, 1)
        # init_ff_layer(self.fc)


    def forward(self, x):
        return T.sigmoid(F.dropout(self.fc(x), self.dropout))


class SimpleMatcher(Module):
    def __init__(self, dim=768, dropout=0.3):
        super(SimpleMatcher, self).__init__()
        self.dim = dim
        # self.fc = Linear(dim, dim)
        self.fcout = Linear(dim*4, 1)
        # init_ff_layer(self.fc)
        # init_ff_layer(self.fcout)
        self.dropout = nn.Dropout(0.3)
        # self.norm = LayerNorm(dim*4)


    def forward(self, a, b):
        c = T.cat([a, b, T.abs(a - b), a * b], dim=-1)
        out = F.sigmoid(F.dropout(self.fcout(c),p=self.dropout))
        return out


class RE2(Module):
    def __init__(self, dim=768, num_block=2):
        super(RE2, self).__init__()
        self.num_block = num_block
        self.blocks = []
        self.predictor = Prediction(dim)
        self.aligner = Aligner(dim)
        for i in range(num_block):
            self.blocks.append(RE2Block(dim, self.aligner))


    def forward(self, emb_a, emb_b):
        o_t2a = T.zeros_like(emb_a, dtype=T.float32)
        o_t1a = T.zeros_like(emb_a, dtype=T.float32)
        o_t2b = T.zeros_like(emb_a, dtype=T.float32)
        o_t1b = T.zeros_like(emb_a, dtype=T.float32)
        assert self.num_block > 0
        for i in range(self.num_block):
            inp_a = T.cat([emb_a, o_t1a + o_t2a], dim=-1)
            inp_b = T.cat([emb_b, o_t1b + o_t2b], dim=-1)
            o_a, o_b = self.blocks[i](inp_a, inp_b)
            o_t2a = o_t1a
            o_t1a = o_a
            o_t2b = o_t1b
            o_t1b = o_b
        y = self.predictor(o_a, o_b)
        return y


class RE2Block(Module):
    def __init__(self, dim, aligner):
        super(RE2Block, self).__init__()
        self.encoder = Encoder(dim)
        self.fuser = Fuser(dim)
        self.aligner = aligner


    def forward(self, inp_a, inp_b):
        encoded_a = self.encoder(inp_a)
        encoded_b = self.encoder(inp_b)
        aligned_a = self.aligner(encoded_a, encoded_b)
        fused_a = self.fuser(encoded_a, aligned_a)
        aligned_b = self.aligner(encoded_b, encoded_a)
        fused_b = self.fuser(encoded_b, aligned_b)
        return fused_a, fused_b


class Encoder(Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.bilstm = LSTM(dim * 2, dim, bidirectional=True, dropout=0.2, num_layers=1)


    def forward(self, x):
        t_h, (_, _) = self.bilstm(x)
        o = T.cat([x, t_h], dim=-1)
        return o


class Fuser(Module):
    def __init__(self, dim):
        super(Fuser, self).__init__()
        self.G1 = Linear(dim * 8, dim)
        # init_ff_layer(self.G1)
        self.G2 = Linear(dim * 8, dim)
        self.G3 = Linear(dim * 8, dim)
        self.G = Linear(dim * 3, dim)
        # init_ff_layer(self.G2)
        # init_ff_layer(self.G3)
        # init_ff_layer(self.G)


    def _fuse(self, z, z_new):
        z1 = self.G1(T.cat([z, z_new], dim=-1))
        z2 = self.G2(T.cat([z, z - z_new], dim=-1))
        z3 = self.G3(T.cat([z, z * z_new], dim=-1))
        z_o = F.dropout(self.G(T.cat([z1, z2, z3], dim=-1)), p=0.2)
        return z_o

    def forward(self, z, z_new):
        z = self._fuse(z, z_new)
        return z


class Aligner(Module):
    def __init__(self, dim):
        super(Aligner, self).__init__()
        self.ff = Linear(dim * 4, dim * 4)
        # init_ff_layer(self.ff)


    def forward(self, ix, iother):
        ex = F.dropout(self.ff(ix), p=0.2)
        eother = F.dropout(self.ff(iother), p=0.2)
        align_x = ex.bmm(eother.transpose(1, 2))
        align_x = F.softmax(align_x, dim=-1)
        aligned = align_x.bmm(iother)
        return aligned


class Prediction(Module):
    def __init__(self, dim):
        super(Prediction, self).__init__()
        self.H = Linear(dim * 4, 1)
        # init_ff_layer(self.H)


    def forward(self, a, b):
        a = T.max(a, dim=1)[0]
        b = T.max(b, dim=1)[0]
        y = F.dropout(self.H(T.cat([a, b, T.abs(a - b), a * b], dim=-1)), p=0.5)
        y = T.sigmoid(y)
        return y


if __name__ == '__main__':
    a = T.ones(32, 768, dtype=T.float32)
    b = T.ones(32, 768, dtype=T.float32)
    y = EuclideanMatcher()(a,b)
    print(y.size())
