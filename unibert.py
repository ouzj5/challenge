import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from einops import rearrange
from category_id_map import CATEGORY_ID_LIST
from other import TransformerModel, MutiSelfAttentionFusion, AFF
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead, BertPooler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    # inuput frame
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
        '''
        text_emb = self.embeddings(input_ids=text_input_ids)
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        # reduce frame feature dimensions : 1536 -> 1024
        # video_feature = self.video_fc(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        '''

        text_emb = self.embeddings(input_ids=text_input_ids)
        video_emb = self.video_fc(video_feature)
        video_emb = gelu(video_emb)
        # video_emb = self.video_embeddings(inputs_embeds=video_emb)

        embedding_output = torch.cat([text_emb, video_emb], 1)
        mask = torch.cat([text_mask, video_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=mask) # ['last_hidden_state']

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return pooled_output

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))