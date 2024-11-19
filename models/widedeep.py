import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

class WideDeep(BaseRanker):

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

        self.linear = ctr.LinearLayer(ctr_fields, dataset)
        self.mlp = MLPModule(
                        [d]+model_config['mlp_layer']+[1],
                        activation_func = model_config['activation'],
                        dropout = model_config['dropout'],
                        batch_norm = model_config['batch_norm'],
                        last_activation = False, last_bn=False)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        wide_score = self.linear(batch)
        emb = self.get_embedddings(batch)
        deep_score = self.mlp(emb).squeeze(-1)
        score = wide_score + deep_score
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
