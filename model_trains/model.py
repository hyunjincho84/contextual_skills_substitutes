# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from transformers.activations import ACT2FN

class BERTForSkillPrediction(nn.Module):
    """
    - BERT 출력의 [MASK] 토큰 위치 벡터만 뽑아 분류하는 구조
    - 주 모델과 동일: Dense -> activation -> LayerNorm -> Linear(NumSkills)
    """
    def __init__(self, model_name_or_path: str, num_skills: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name_or_path)
        hidden = self.bert.config.hidden_size
        self.dense = nn.Linear(hidden, hidden)
        self.transform_act_fn = ACT2FN[self.bert.config.hidden_act]
        self.layer_norm = nn.LayerNorm(hidden, eps=self.bert.config.layer_norm_eps)
        self.classifier = nn.Linear(hidden, num_skills)

    def forward(self, input_ids, attention_mask, mask_idx):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        mask_idx: [B]  (각 배치에서 [MASK]가 있는 위치 인덱스)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        mask_vec = last_hidden[torch.arange(input_ids.size(0), device=input_ids.device), mask_idx]  # [B, H]
        x = self.dense(mask_vec)
        x = self.transform_act_fn(x)
        x = self.layer_norm(x)
        logits = self.classifier(x)  # [B, num_skills]
        return logits