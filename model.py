import torch
import torch.autograd as autograd
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AlbertModel, AlbertConfig
import torch.nn.functional as F
import math


class BERT_rank(nn.Module):
    def __init__(self, out_dim=1, dropout=0.1):
        super(BERT_rank, self).__init__()
        self.bert_config = BertConfig.from_pretrained('../bert-base-uncased/config.json')
        self.bert = BertModel.from_pretrained('../bert-base-uncased/pytorch_model.bin', config=self.bert_config)
        self.out_dim = out_dim
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(768, self.out_dim)
        self.hide1 = nn.Linear(out_dim * 3, out_dim)
        self.hide2 = nn.Linear(out_dim, out_dim)
        self.out = nn.Linear(out_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.ff1 = nn.Linear(10, 10)
        self.ff2 = nn.Linear(10, 1)

    def forward(self, pair_sentence):
        # context_output = self.bert(artcile_input_ids, article_input_mask)[0]  # (batch x len x dim)
        # context_output = torch.mean(context_output[0], 1)
        # context_output = self.linear(context_output)

        # title_output = self.bert(title_input_ids, title_input_mask)[0]  # (batch x len x dim)
        # title_output = torch.mean(title_output[0], 1)
        # title_output = self.linear(title_output)
        # return self.scoring(context_output, article_input_mask, title_output, title_input_mask)
        output = self.encode(pair_sentence)
        score = self.scoring(output)
        return score

    def encode(self, pair_sentence):
        # context_output = self.bert(context_input_ids, context_input_mask)[0]
        # # context_output = torch.mean(context_output[0], 1)
        # context_output = self.linear(context_output)
        # return context_output, context_input_mask
        output = self.bert(pair_sentence[0], pair_sentence[2], pair_sentence[1])[1]
        return output

    def scoring(self, output):
        # output_abs = torch.abs(text1 - text2)
        # target_span_embedding = torch.cat((text1, text2, output_abs), dim=1)
        #
        # hide_1 = F.relu(self.hide1(target_span_embedding))
        # hide_2 = self.dropout_layer(hide_1)
        # hide = F.relu(self.hide2(hide_2))
        # output = self.out(hide)
        # score = self.sigmoid(output)

        # article_len = text1.shape[1]
        # title_len = text2.shape[1]
        # c = text1.unsqueeze(dim=2)
        # c = c.repeat([1, 1, title_len, 1])
        # q = text2.unsqueeze(dim=1)
        # q = q.repeat([1, article_len, 1, 1])
        # s = q - c
        # s = torch.sum(s * s, dim=3)
        # s = torch.exp(-1 * s)
        # text1_mask = text1_mask.unsqueeze(dim=2)
        # text1_mask = text1_mask.repeat([1, 1, title_len])
        # text2_mask = text2_mask.unsqueeze(dim=1)
        # text2_mask = text2_mask.repeat([1, article_len, 1])
        # s = s * text1_mask * text2_mask
        # row_max, _ = torch.max(s, dim=2)
        # row_max, _ = row_max.topk(10, dim=1, largest=True)
        # row_max = F.relu(self.ff1(row_max))
        # row_max = self.ff2(row_max)
        # score = row_max.squeeze()
        score = self.linear(output)
        score = self.sigmoid(self.dropout_layer(score))
        return score.squeeze()


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_score, retrieve_score):
        loss = torch.log(1+torch.exp(-(pos_score-retrieve_score)))
        loss = loss.mean()
        return loss
