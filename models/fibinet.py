import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

r"""
FiBiNET
######################

Paper Reference:
    FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction (RecSys'19)
    https://dl.acm.org/doi/abs/10.1145/3298689.3347043
"""

class FiBiNET(BaseRanker):

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
        self.senet = ctr.SqueezeExcitation(
                        d // self.embed_dim, 
                        model_config['reduction_ratio'], 
                        model_config['excitation_activation'])
        self.bilinear = ctr.BilinearInteraction(
                            d // self.embed_dim, 
                            self.embed_dim, 
                            model_config['bilinear_type'])
        if not model_config['shared_bilinear']:
            self.bilinear4se = ctr.BilinearInteraction(
                                d // self.embed_dim, 
                                self.embed_dim, 
                                model_config['bilinear_type'])
        self.mlp = MLPModule(
                        [d // self.embed_dim * (d // self.embed_dim - 1) * self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'], 
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'],
                        last_activation=False, 
                        last_bn=False)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        lr_score = self.linear(batch)
        emb = self.get_embedddings(batch)
        emb = emb.reshape(emb.shape[0], -1, self.embed_dim)
        senet_emb = self.senet(emb)
        bilinear_ori = self.bilinear(emb)
        if self.config_dict['ctr_model']['shared_bilinear']:
            bilinear_senet = self.bilinear(senet_emb)
        else:
            bilinear_senet = self.bilinear4se(senet_emb)
        comb = torch.cat([bilinear_ori, bilinear_senet], dim=1)
        mlp_score = self.mlp(comb.flatten(1)).squeeze(-1)
        score = lr_score + mlp_score
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
