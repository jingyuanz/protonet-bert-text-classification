from torch import nn
import numpy as np
from torch.nn import *

def create_emb_layer(weights_matrix, non_trainable=False):
    print(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


class BILSTM_encoder(nn.Module):
    def __init__(self, out_dim=300, in_dim=300, dropout=0.5, embedding_matrix=None, vocab_size=10000):
        super(BILSTM_encoder, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.dropout = dropout
        if not embedding_matrix:
            embedding_matrix = np.zeros((vocab_size, in_dim))
        self.emb_layer = create_emb_layer(embedding_matrix)
        self.bilstm = LSTM(in_dim, out_dim, bidirectional=True, dropout=0.2, num_layers=1)

    def load_pretrained_embedding(self, embedding_matrix):
        self.emb_layer = create_emb_layer(embedding_matrix)

    def forward(self, x):
        x = self.emb_layer(x)
        t_h, (_, _) = self.bilstm(x)
        return t_h, t_h[-1]




