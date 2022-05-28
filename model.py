import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from einops import rearrange
from category_id_map import CATEGORY_ID_LIST
from other import TransformerModel, MutiSelfAttentionFusion, AFF

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        bert_output_size = 768
        bert_input_size = 512
        # TODO add attention
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        # self.attention = SelfAttention(1, 1, 1)


        # TODO replace the concatDense
        # v2 add residual PreNorm

        out_dim = 1
        embedding_dim = 1536
        num_heads = 4
        num_layers = 4
        hidden_dim = 512    #

        # transformer fusion
        # self.video_to_bert = nn.Linear(args.vlad_hidden_size, bert_output_size)
        # self.fusion = TransformerModel(
        #     embedding_dim,
        #     num_layers,
        #     num_heads,
        #     hidden_dim,
        #     args.vlad_hidden_size, bert_output_size, args.dropout, args.fc_size
        # )

        # attention
        self.fusion = MutiSelfAttentionFusion(1, args.vlad_hidden_size, bert_output_size, args.dropout, args.fc_size)

        # baseline
        # self.fusion = ConcatDenseSE( args.vlad_hidden_size + bert_output_size, args.fc_size, args.dropout, args.se_ratio)

        # AFF
        # self.video_to_bert = nn.Linear(args.vlad_hidden_size, bert_output_size)
        # self.fusion = AFF(bert_output_size * 2, 16)
        self.to_fc = nn.Linear(bert_output_size, args.fc_size)

        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):

        bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['pooler_output']

        vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # TODO add attention
        vision_embedding = self.enhance(vision_embedding)

        # vision_embedding = self.attention(vision_embedding)

        # TODO replace the concatDense
        # final_embedding = self.fusion([vision_embedding, bert_embedding]) # baseline

        # transformer fusion
        # vision_embedding = self.video_to_bert(vision_embedding)
        # sum_embedding = torch.cat([bert_embedding, vision_embedding], 1)
        # final_embedding = self.fusion(sum_embedding)

        # attention fusion
        # sum_embedding = torch.cat([bert_embedding, vision_embedding], 1)
        # exchange pos
        sum_embedding = torch.cat([vision_embedding, bert_embedding], 1)
        # final_embedding = self.fusion(sum_embedding)

        # AFF
        # vision_embedding = self.video_to_bert(vision_embedding)
        # sum_embedding = torch.cat([vision_embedding, bert_embedding], 1)
        # final_embedding = self.fusion(sum_embedding, sum_embedding)
        # final_embedding = self.to_fc(final_embedding)

        final_embedding = self.to_fc(vision_embedding)
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x



class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        # self.dim_q = dim_q
        # self.dim_k = dim_k
        # self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, 1
        # 根据文本获得相应的维度
        x = torch.unsqueeze(x, 2)
        batch, n, dim_q = x.shape
        # assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, dim_k
        k = self.linear_k(x)  # batch, dim_k
        v = self.linear_v(x)  # batch, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att.squeeze(2)


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, dropout, se_ratio):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        # TODO add attention
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)
        # self.attention = SelfAttention(1, 1, 1)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        # TODO add attention
        embedding = self.enhance(embedding)
        # embedding = self.attention(embedding)

        return embedding
