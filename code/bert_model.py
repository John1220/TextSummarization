# -*- coding:utf-8 -*-
""" """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertEdgeScorer(nn.Module):
    """BERT model for computing sentence similarity and scoring edges.

    """

    def __init__(self, config):
        super(BertEdgeScorer, self).__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, input_ids_c, token_type_ids_c, attention_mask_c,
                labels=None):
        # inputs_id: B * T
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        # pooled_output: batch_size * hidden_size
        sequence_output, pooled_output_c = self.bert(input_ids_c, attention_mask_c, token_type_ids_c)
        logits = torch.bmm(pooled_output.unsqueeze(1), pooled_output_c.unsqueeze(2)).view(-1)
        pros = torch.sigmoid(logits)

        return logits, pros
