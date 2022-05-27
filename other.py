import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from einops import rearrange
from category_id_map import CATEGORY_ID_LIST

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, vlad_hidden_size, bert_output_size, dropout, fc_size):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(PreNorm(dim, MutiSelfAttention(1, vlad_hidden_size, bert_output_size, dropout, fc_size)), dropout),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim)), dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.fusion_dropout = nn.Dropout(dropout)
        self.cl_mlp = nn.Linear(bert_output_size * 2, fc_size)
    def forward(self, x):
        x = self.net(x)
        x = self.fusion_dropout(x)
        x = self.cl_mlp(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn, dropout):
        super().__init__()
        self.fn = fn
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout_layer(self.fn(x)) + x

class MutiSelfAttention(nn.Module):
    def __init__(self, dim, vald_size, bert_size, dropout, output_size, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = bert_size ** -0.5
        self.dim = dim
        self.output_size = output_size
        self.fusion_dropout = nn.Dropout(dropout)
        self.bert_size = bert_size
        self.vald_size = vald_size

        self.to_qkv1 = nn.Linear(bert_size, bert_size * 3, bias=False)
        self.to_q2 = nn.Linear(bert_size, bert_size, bias=False)
        self.to_out = nn.Linear(2 * bert_size, bert_size) # 两个q合并了，维度乘2

    def forward(self, x):   # x[0]: batch_size, feature num
        b = x[0].shape
        h = self.heads
        x1 = x[:, :self.bert_size]    # bert feature
        qkv = self.to_qkv1(x1)
        q, k, v = rearrange(qkv, 'b (qkv h d) -> qkv b h d', qkv=3, h=h)  # h: head t: feature num   d:feat vector

        x2 = x[:, self.bert_size:]    # video featrue
        qkv = self.to_q2(x2)
        q2 = rearrange(qkv, 'b (qkv h d) -> qkv b h d', qkv=1, h=h)

        q2 = q2.squeeze(-4) # 在头部加个qkv维度

        q = torch.cat((q, q2), axis=2)

        dots = torch.einsum('bhid,bhjd->bhij', q.unsqueeze(3), k.unsqueeze(3)) * self.scale # 算相关度
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v.unsqueeze(3))
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = torch.flatten(out, 1)

        # v2 add drop out
        out = self.fusion_dropout(out)
        # out = self.to_out(out)
        return out

class MutiSelfAttentionFusion(nn.Module):
    def __init__(self, dim, vald_size, bert_size, dropout, output_size, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = bert_size ** -0.5
        self.dim = dim
        self.output_size = output_size
        self.fusion_dropout = nn.Dropout(dropout)
        self.bert_size = bert_size
        self.vald_size = vald_size

        self.to_qkv1 = nn.Linear(bert_size, output_size * 3, bias=False)
        self.to_q2 = nn.Linear(vald_size, output_size, bias=False)
        self.to_out = nn.Linear(2 * output_size, output_size) # 两个q合并了，维度乘2

    def forward(self, x):   # x[0]: batch_size, feature num
        b = x[0].shape
        h = self.heads
        x1 = x[:, :self.bert_size]    # bert feature
        qkv = self.to_qkv1(x1)
        q, k, v = rearrange(qkv, 'b (qkv h d) -> qkv b h d', qkv=3, h=h)  # h: head t: feature num   d:feat vector

        x2 = x[:, self.bert_size:]    # video featrue
        qkv = self.to_q2(x2)
        q2 = rearrange(qkv, 'b (qkv h d) -> qkv b h d', qkv=1, h=h)

        q2 = q2.squeeze(-4) # 在头部加个qkv维度

        q = torch.cat((q, q2), axis=2)

        dots = torch.einsum('bhid,bhjd->bhij', q.unsqueeze(3), k.unsqueeze(3)) * self.scale # 算相关度
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v.unsqueeze(3))
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = torch.flatten(out, 1)

        # v2 add drop out
        out = self.fusion_dropout(out)
        out = self.to_out(out)
        return out