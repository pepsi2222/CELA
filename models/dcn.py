import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

r"""
DCN
######################

Paper Reference:
    Deep & Cross Network for Ad Click Predictions (ADKDD'17)
    https://dl.acm.org/doi/10.1145/3124749.3124754
"""

class DCN(BaseRanker):

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

        self.cross_net = ctr.CrossNetwork(d, model_config['num_layers'])
        self.mlp = MLPModule(
                    [d] + model_config['mlp_layer'],
                    model_config['activation'],
                    model_config['dropout'],
                    batch_norm=model_config['batch_norm'])
        self.fc = torch.nn.Linear(d + model_config['mlp_layer'][-1], 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        emb = self.get_embedddings(batch)
        cross_out = self.cross_net(emb)
        deep_out = self.mlp(emb)
        score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
