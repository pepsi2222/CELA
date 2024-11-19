import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

r"""
PNN
######################

Paper Reference:
    Product-based Neural Networks for User Response Prediction (ICDM'16)
    https://ieeexplore.ieee.org/document/7837964
"""

class PNN(BaseRanker):

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

        num_fields = d // self.embed_dim
        if model_config['stack_dim'] is None:
            if model_config['product_type'].lower() == 'inner':
                self.prod_layer = ctr.InnerProductLayer(num_fields)
                mlp_in_dim = num_fields * (num_fields - 1) // 2 + num_fields * self.embed_dim
            elif model_config['product_type'].lower() == 'outer':
                self.prod_layer = ctr.OuterProductLayer(num_fields)
                mlp_in_dim = (num_fields * (num_fields - 1) // 2) * self.embed_dim * self.embed_dim + num_fields * self.embed_dim
            else:
                raise ValueError(f'Expect product_type to be `inner` or `outer`, but got {model_config["product_type"]}.')
        else:
            self.Wz = nn.Parameter(torch.randn(num_fields * self.embed_dim, model_config['stack_dim']))
            if model_config['product_type'].lower() == 'inner':
                self.Thetap = nn.Parameter(torch.randn(num_fields, model_config['stack_dim']))
            elif model_config['product_type'].lower() == 'outer':
                self.Wp = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim, model_config['stack_dim']))
            else:
                raise ValueError(f'Expect product_type to be `inner` or `outer`, but got {model_config["product_type"]}.')
            self.bias = nn.Parameter(torch.randn(model_config['stack_dim']))
            mlp_in_dim = 2 * model_config['stack_dim']
            
        self.mlp = MLPModule(
                    [mlp_in_dim] + model_config['mlp_layer'] + [1],
                    model_config['activation'], model_config['dropout'],
                    batch_norm=model_config['batch_norm'],
                    last_activation=False, last_bn=False)


        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        emb = self.get_embedddings(batch)
        emb = emb.reshape(emb.shape[0], -1, self.embed_dim)
        if self.config_dict['ctr_model']['stack_dim'] is None:
            lz = emb.flatten(1)                                             # B x F*D
            lp = self.prod_layer(emb)                                       # B x num_pairs
        else:
            lz = (self.Wz * emb.view(emb.size(0), -1, 1)).sum(1)            # B x S
            if self.config_dict['ctr_model']['product_type'] == 'inner':
                delta = torch.einsum('fs,bfd->bfsd', [self.Thetap, emb])    # B x F x S x D
                lp = (delta.sum(1)**2).sum(-1)                              # B x S
            elif self.config_dict['ctr_model']['product_type'] == 'outer':
                p = torch.einsum('bi,bj->bij', 2 * [emb.sum(1)])            # B x D x D
                lp = torch.einsum('bij,ijs->bs', [p, self.Wp])              # B x S
        mlp_in = torch.cat([lz, lp], dim=1)
        score = self.mlp(mlp_in).squeeze(-1)
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
