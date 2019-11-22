import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from bert import BertPreTrainedModel, BertModel

from batch_loader import WHITESPACE_PLACEHOLDER

class KBQA(BertPreTrainedModel):
    def __init__(self, config):
        super(KBQA, self).__init__(config)
        self.bert = BertModel(config)
        self.ner_layer = nn.Linear(config.hidden_size, 2) # head, tail
        self.re_layer = nn.Linear(config.hidden_size, 1) # yes/no
        self.apply(self.init_bert_weights)

    def forward(self, batch_data, task_id, is_train):
        if task_id == 0: # task is ner
            token_ids, token_types, head, tail = batch_data
            attention_mask = token_ids.gt(0)
            sequence_output, _ = self.bert(token_ids, token_types, attention_mask, output_all_encoded_layers=False)
            head_logits, tail_logits = self.ner_layer(sequence_output).split(1, dim=-1)
            head_logits = head_logits.squeeze(dim=-1)
            tail_logits = tail_logits.squeeze(dim=-1)
            logits = (head_logits, tail_logits)
            if is_train:
                seq_lengths = attention_mask.sum(-1).float()
                ignored_index = head_logits.size(1)
                head.clamp_(0, ignored_index)
                tail.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                head_loss = loss_fct(head_logits, head)
                tail_loss = loss_fct(tail_logits, tail)
                loss = (head_loss + tail_loss) / 2
        else: # task is re
            token_ids, token_types, label = batch_data
            attention_mask = token_ids.gt(0)
            _, pooled_output = self.bert(token_ids, token_types, attention_mask, output_all_encoded_layers=False)
            logits = self.re_layer(pooled_output).squeeze(-1)
            if is_train:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, label)
        return loss if is_train else logits