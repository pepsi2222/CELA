import math
import torch
import torch.nn as nn
from .baseranker import BaseRanker
from RecStudio.recstudio.model.module.layers import AttentionLayer, MLPModule


r"""
DIN
######################

Paper Reference:
    Guorui Zhou, et al. "Deep Interest Network for Click-Through Rate Prediction" in KDD2018.
    https://dl.acm.org/doi/10.1145/3219819.3219823

"""


class DIN(BaseRanker):
    r"""
        | Deep Interest Network (DIN) designs a local activation unit to adaptively learn the representation
          of user interests from historical behaviors with respect to a certain ad.

        | DIN calculate the relevance between the target item and items in the sequence by adapting an
          attention machnism. But the method could not be applied to recall on all items in prediction
          due to the huge time cost.
    """


    def __init__(self, config, dataset, ctr_fields, item_text_field=None, **kwargs):
        super().__init__(config, dataset, ctr_fields, item_text_field, **kwargs)

        d = self.embed_dim * len([_ for _ in self.item_fields if dataset.field2type[_] != 'text'])
        if item_text_field is not None:
            d += self.embed_dim

        
        model_config = self.config_dict['ctr_model']
        self.item_bias = nn.Embedding(dataset.num_items, 1, padding_idx=0)
        self.activation_unit = AttentionLayer(
                                3*d, 
                                d, 
                                mlp_layers=model_config['attention_mlp'], 
                                activation=model_config['activation'])
        norm = [nn.BatchNorm1d(d)] if model_config['batch_norm'] else []
        norm.append(nn.Linear(d, d))
        self.norm = nn.Sequential(*norm)

        dense_mlp_in_dim = 3*d + self.embed_dim * (len(self.user_fields) + len(self.context_fields))
        self.dense_mlp = MLPModule(
                            [dense_mlp_in_dim]+model_config['fc_mlp'], 
                            activation_func=model_config['activation'], 
                            dropout=model_config['dropout'], 
                            batch_norm=model_config['batch_norm'])
        self.fc = nn.Linear(model_config['fc_mlp'][-1], 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.init_parameters()


    def forward(self, **batch):
        seq_emb = self.get_behavior_embeddings(batch).tile(math.ceil(len(self.item_fields) / len(self.behavior_fields)))
        target_emb = self.get_item_embeddings(batch)
        item_bias = self.item_bias(batch[self.fiid]).squeeze(-1)

        key_padding_mask = (batch['in_'+self.fiid] == 0)
        key_padding_mask[:, 0] = False
        target_emb_ = target_emb.unsqueeze(1).repeat(1, seq_emb.size(1), 1)
        attn_seq = self.activation_unit(
            query=target_emb.unsqueeze(1),
            key=torch.cat((target_emb_, target_emb_*seq_emb, target_emb_-seq_emb), dim=-1),
            value=seq_emb,
            key_padding_mask=key_padding_mask,
            softmax=True
        ).squeeze(1)
        attn_seq = self.norm(attn_seq)

        cat_emb = [attn_seq, target_emb, target_emb*attn_seq]
        if len(self.user_fields) > 0:
            user_emb = self.get_user_embeddings(batch)
            cat_emb.append(user_emb)
        if len(self.context_fields) > 0:
            context_emb = self.get_context_embeddings(batch)
            cat_emb.append(context_emb)
        cat_emb = torch.cat(cat_emb, dim=-1)
        
        score = self.fc(self.dense_mlp(cat_emb)).squeeze(-1)
        score += item_bias
        loss = self.loss_fn(score, batch[self.frating])
        return {
            'loss': loss,
            'logits': score
        }
