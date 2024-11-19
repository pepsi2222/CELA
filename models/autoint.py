import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import AttentionLayer, MLPModule

r"""
AutoInt
######################

Paper Reference:
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (CIKM'19)
    https://dl.acm.org/doi/abs/10.1145/3357384.3357925
"""


class AutoInt(BaseRanker):

    def __init__(self, config, dataset, ctr_fields, item_text_field=None, **kwargs):
        super().__init__(config, dataset, ctr_fields, item_text_field, **kwargs)

        model_config = self.config_dict['ctr_model']

        d = self.embed_dim * (
                    len([_ for _ in self.item_fields if dataset.field2type[_] != 'text']) + \
                    len([_ for _ in self.behavior_fields if dataset.field2type[_.replace('in_', '')] != 'text']) + \
                    len(self.user_fields) + \
                    len(self.context_fields)
                )
        if item_text_field is not None:
            d += self.embed_dim * 2

        if model_config['wide']:
            self.linear = ctr.LinearLayer(ctr_fields, dataset)
        if model_config['deep']:
            self.mlp = MLPModule([d] + model_config['mlp_layer'] + [1],
                            model_config['activation'], 
                            model_config['dropout'],
                            batch_norm=model_config['batch_norm'],
                            last_activation=False, 
                            last_bn=False
                        )
        self.int = nn.Sequential(*[
                        ctr.SelfAttentionInteractingLayer(
                            self.embed_dim if i == 0 else model_config['attention_dim'],
                            activation=model_config['activation'],
                            n_head=model_config['n_head'],
                            dropout=model_config['dropout'],
                            residual=model_config['residual'],
                            residual_project=model_config['residual_project'],
                            layer_norm=model_config['layer_norm']
                        )
                        for i in range(model_config['num_attention_layers'])])
        self.fc = nn.Linear(d, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()


    def forward(self, **batch):
        emb = self.get_embedddings(batch)
        emb = emb.reshape(emb.shape[0], -1, self.embed_dim)
        attn_out = self.int(emb)
        int_score = self.fc(attn_out.flatten(1)).squeeze(-1)
        score = int_score
        if self.config_dict['ctr_model']['wide']:
            lr_score = self.linear(batch)
            score += lr_score
        if self.config_dict['ctr_model']['deep']:
            mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
            score += mlp_score
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
