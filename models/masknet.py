from torch import nn 
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module import ctr

r"""
MaskNet
######################

Paper Reference:
    MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask (DLP KDD'21)
    https://arxiv.org/abs/2102.07619
"""

class MaskNet(BaseRanker):

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

        if model_config['parallel']:
            self.masknet = ctr.ParallelMaskNet(
                            d // self.embed_dim, 
                            self.embed_dim, 
                            model_config['num_blocks'], 
                            model_config['block_dim'], 
                            model_config['reduction_ratio'],
                            model_config['mlp_layer'],
                            model_config['activation'],
                            model_config['dropout'],
                            model_config['hidden_layer_norm'])
        else:
            self.masknet = ctr.SerialMaskNet(
                            d // self.embed_dim, 
                            self.embed_dim, 
                            model_config['block_dim'], 
                            model_config['reduction_ratio'],
                            model_config['activation'],
                            model_config['dropout'],
                            model_config['hidden_layer_norm'])
            
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()


    def forward(self, **batch):
        emb = self.get_embedddings(batch) # [B, D]
        emb = emb.reshape(emb.shape[0], -1, self.embed_dim)
        score = self.masknet(emb).squeeze(-1)
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
