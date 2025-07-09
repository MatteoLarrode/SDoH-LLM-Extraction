import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel

class RobertaBinaryClassifierWithWeight(RobertaPreTrainedModel):
    def __init__(self, config, pos_weight):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])  # CLS token
        logits = self.classifier(pooled_output)

        if labels is not None:
            labels = labels.unsqueeze(1)  # shape: [batch_size, 1]
            loss = self.loss_fct(logits, labels)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}