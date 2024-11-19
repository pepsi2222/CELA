import os
import torch
import torch.nn as nn
from module.loss import *
from RecStudio.recstudio.model.module.ctr import MLPModule
from utils.utils import ModelState
from utils import ENCODER_TYPE_MODELS, DECODER_TYPE_MODELS

CTR_MODEL_WEIGHTS_NAME = 'ctr_model.bin'

class Align(nn.Module):
    """
    Align language model with ctr model.
    """
    def __init__(self, config, language_model, ctr_model, item_text_field, ctr_fields):
        super().__init__()
        self.config_dict = config
        self.language_model = language_model
        self.ctr_model = ctr_model
        self.item_text_field = item_text_field
        self.ctr_fields = ctr_fields

        if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
            lm_proj_in_dim = language_model.config.hidden_size
        elif self.language_model.config.model_type in DECODER_TYPE_MODELS:
            lm_proj_in_dim = language_model.config.word_embed_proj_dim
        else:
            raise ValueError(f'language_model should be in {"|".join(ENCODER_TYPE_MODELS + DECODER_TYPE_MODELS)}, " \
                             "but got {self.language_model.config.model_type}.')

        if config['ctr_model']['projection_layers'] is not None:
            assert config['ctr_model']['projection_layers'][-1] == config['language_model']['projection_layers'][-1], \
                    'the last layer in `projection_layers` of ctr_model and language_model should be the same.'
            self.cm_projection = nn.Sequential(
                                    MLPModule(
                                        mlp_layers=[ctr_model.embed_dim * (len(ctr_model.item_fields) - 1)] + \
                                                    config['ctr_model']['projection_layers'],
                                        activation_func=config['ctr_model']['activation'],
                                        dropout=config['ctr_model']['dropout']
                                    ),
                                    nn.LayerNorm(config['ctr_model']['projection_layers'][-1])
                                )
            self.lm_projection = nn.Sequential(
                                    MLPModule(
                                        mlp_layers=[lm_proj_in_dim] + \
                                                    config['language_model']['projection_layers'],
                                        activation_func=config['language_model']['activation'],
                                        dropout=config['language_model']['dropout']
                                    ),
                                    nn.LayerNorm(config['language_model']['projection_layers'][-1])
                                )
        else:
            lm_proj_out_dim = ctr_model.embed_dim * (len(ctr_model.item_fields) - 1)
            self.lm_projection = nn.Sequential(
                                    MLPModule(
                                        mlp_layers=[lm_proj_in_dim, lm_proj_out_dim],
                                        activation_func=config['language_model']['activation'],
                                        dropout=config['language_model']['dropout']
                                    ),
                                    nn.LayerNorm(lm_proj_out_dim)
                                )
        self.loss_fn = self._get_loss_fn()

        self.lm_state = ModelState.ON
        self.ctr_state = ModelState.ON


    def forward(self, return_loss=True, **batch):
        text_emb = self.get_item_text_embeddings(batch)

        if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
            text_proj = self.lm_projection(text_emb)    # B x D1 -> B x D2
        else:
            text_proj = self.lm_projection(text_emb)    # B x L x D1 -> B x L x D2
            bs, sequence_lengths = text_proj.shape[:2]
            sequence_lengths = (torch.eq(
                                        self._get_item_text_feat(batch)['input_ids'], 
                                        self.language_model.config.pad_token_id
                                    ).long().argmax(-1) - 1
                                ).to(text_proj.device)
            text_proj = text_proj[torch.arange(bs, device=text_proj.device), sequence_lengths]    # B x D2

        ctr_emb = self.get_item_ctr_embeddings(batch)
        if self.config_dict['ctr_model']['projection_layers'] is not None:
            ctr_proj = self.cm_projection(ctr_emb)
        else:
            ctr_proj = ctr_emb

        if not self.config_dict['loss']['weighted_by_item_popularity']:
            loss = self.loss_fn(text_proj, ctr_proj)
        else:
            popularity = batch['popularity']
            # version 1
            # loss = torch.mean(popularity * self.loss_fn(text_proj, ctr_proj))
            # version 2
            loss = (popularity * self.loss_fn(text_proj, ctr_proj)).sum() / popularity.sum()
            # version 3
            loss = torch.sum(popularity * self.loss_fn(text_proj, ctr_proj))
        # if not self.config_dict['loss']['simcse']:
        #     loss = self.loss_fn(text_proj, ctr_proj)
        # else:
        #     text_emb_2 = self.get_item_text_embeddings(batch)
        #     text_proj_2 = self.lm_projection(text_emb_2)
        #     loss = self.loss_fn(text_proj, ctr_proj, text_proj_2)
        return {
            'loss': loss
        }
    
    def _get_loss_fn(self):
        loss_fn = self.config_dict['loss']['loss_fn']
        reduction = 'none' if self.config_dict['loss']['weighted_by_item_popularity'] else 'mean'

        if loss_fn.lower() == 'mse':
            return nn.MSELoss(reduction=reduction)
        elif loss_fn.lower() == 'bpr':
            return BPRLoss(self.config_dict['loss']['score_fn'], self.config_dict['loss']['neg_count'], reduction=reduction)
        elif loss_fn.lower() == 'infonce':
            return InfoNCELoss(self.config_dict['loss']['score_fn'], self.config_dict['loss']['temperature'], reduction=reduction)
            # return InfoNCELoss(self.config_dict['loss']['score_fn'], self.config_dict['loss']['temperature'], self.config_dict['loss']['simcse'])
        else:
            raise ValueError('wrong loss_fn')
    

    def _get_item_text_feat(self, batch):
        return batch[self.item_text_field]
    

    def _get_item_ctr_feat(self, batch):
        return dict((field, value) for field, value in batch.items() if field in self.ctr_fields and \
                                                                        field in self.ctr_model.item_fields)
    

    def get_item_text_embeddings(self, batch):
        item_text_feat = self._get_item_text_feat(batch)
        item_text_embs = self.language_model(**item_text_feat, return_dict=True)
        if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
            item_text_embs = item_text_embs.last_hidden_state[:, 0, :]
        else:
            item_text_embs = item_text_embs.last_hidden_state
        return item_text_embs


    def get_item_ctr_embeddings(self, batch):
        item_ctr_feat = self._get_item_ctr_feat(batch)
        item_ctr_embs = self.ctr_model.embedding(item_ctr_feat)
        return item_ctr_embs


    def freeze(self, model_type):
        # projection layer is always waken
        if model_type == 'ctr_model':
            model = self.ctr_model
            self.ctr_state = ModelState.OFF
        else:
            model = self.language_model
            self.lm_state = ModelState.OFF
            
        for n, p in model.named_parameters():
            p.requires_grad = False


    # def wake(self, model_type):
    #     if model_type == 'ctr_model':
    #         model = self.ctr_model
    #         self.ctr_state = ModelState.ON
    #     else:
    #         model = self.language_model
    #         self.lm_state = ModelState.ON

    #     for n, p in model.named_parameters():
    #         p.requires_grad = True


    # def switch_states(self):
    #     """
    #     projection is always awake
    #     """
    #     if self.ctr_state == ModelState.OFF:
    #         self.wake('ctr_model')
    #     else:
    #         self.freeze('ctr_model')

    #     if self.lm_state == ModelState.OFF:
    #         self.wake('language_model')
    #     else:
    #         self.freeze('language_model')


    def save(self, save_dir):
        torch.save(self.ctr_model.state_dict(), os.path.join(save_dir, CTR_MODEL_WEIGHTS_NAME))
        self.language_model.save_pretrained(save_dir, safe_serialization=False)
        

    # def load(self, save_dir):
    #     self.ctr_model.load_state_dict(torch.load(os.path.join(save_dir, CTR_MODEL_WEIGHTS_NAME)))
    #     self.language_model.from_pretrained(...)


