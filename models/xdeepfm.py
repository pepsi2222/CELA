import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr
from RecStudio.recstudio.model.module.layers import MLPModule

class xDeepFM(BaseRanker):

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
                    model_config['activation'], 
                    model_config['dropout'],
                    batch_norm=model_config['batch_norm'],
                    last_activation=False, 
                    last_bn=False)
        self.cin = ctr.CIN(self.embed_dim, d // self.embed_dim,
                           model_config['cin_layer_size'], model_config['activation'],
                           direct=model_config['direct'])

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()
        

    def forward(self, **batch):
        lr_score = self.linear(batch)
        emb = self.get_embedddings(batch)
        cin_score = self.cin(emb.reshape(emb.shape[0], -1, self.embed_dim)).squeeze(-1)
        mlp_score = self.mlp(emb).squeeze(-1)
        score = lr_score + cin_score + mlp_score
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
