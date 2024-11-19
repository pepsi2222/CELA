import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

r"""
DCNv2
######################

Paper Reference:
    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems (WWW'21)
    https://dl.acm.org/doi/10.1145/3442381.3450078
"""

class DCNv2(BaseRanker):

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

        if model_config['low_rank'] is None:
            self.cross_net = ctr.CrossNetworkV2(d, model_config['num_layers'])
        else:
            self.cross_net = ctr.CrossNetworkMix(d, model_config['num_layers'], 
                                                model_config['low_rank'], model_config['num_experts'],
                                                model_config['cross_activation'])
            
        if model_config['combination'].lower() == 'parallel':
            self.mlp = MLPModule(
                        [d] + model_config['mlp_layer'],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'])
            self.fc = nn.Linear(d + model_config['mlp_layer'][-1], 1)
        elif model_config['combination'].lower() == 'stacked':
            self.mlp = MLPModule(
                        [d] + model_config['mlp_layer'] + [1],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'],
                        last_activation=False,
                        last_bn=False)
        else:
            raise ValueError(f'Expect combination to be `parallel`|`stacked`, but got {model_config["combination"]}.')

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        emb = self.get_embedddings(batch)
        cross_out = self.cross_net(emb)
        if self.config_dict['ctr_model']['combination'].lower() == 'parallel':
            deep_out = self.mlp(emb)
            score = self.fc(torch.cat([deep_out, cross_out], -1)).squeeze(-1)
        else:
            deep_out = self.mlp(cross_out)
            score = deep_out.squeeze(-1)
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
