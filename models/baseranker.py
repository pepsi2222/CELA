import os
import numpy as np
import torch
import torch.nn as nn
from module.ctr import MyEmbeddings
from transformers.utils import WEIGHTS_NAME
from module.trainer import ALIGN_WEIGHTS_NAME
from RecStudio.recstudio.model.module.ctr import MLPModule
from RecStudio.recstudio.model import init
from collections import defaultdict
import loralib as lora
from utils.utils import ModelState, FROZEN_TEXT_EMBEDDINGS_NAME
from tqdm import tqdm
from utils import ENCODER_TYPE_MODELS, DECODER_TYPE_MODELS

class BaseRanker(nn.Module):
    def __init__(self, config, dataset, ctr_fields: set, item_text_field=None, 
                 language_model=None,  item_text_feat=None, tokenizer=None,
                 lm_opt_strategy='all', lm_projection_group='ctr_model',
                 text_embedding=None):
        super().__init__()
        self.config_dict = config
        self.embed_dim = config['ctr_model']['embed_dim']

        self.fuid = dataset.fuid
        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.user_fields = set(dataset.user_feat.fields) if dataset.name != 'app_gallery' else set(dataset.user_fields)
        self.item_fields = set(dataset.item_feat.fields)
        self.context_fields = set() if dataset.name != 'app_gallery' else set(dataset.context_fields)
        self.behavior_fields = {'in_'+_ for _ in self.item_fields}  if dataset.name != 'app_gallery' \
                                                                    else {f'in_{self.fiid}'} if item_text_field is None \
                                                                    else {f'in_{self.fiid}', item_text_field}

        self.ctr_fields = ctr_fields
        self.item_text_field = item_text_field
        self.language_model = language_model
        self.item_text_feat = item_text_feat
        self.tokenizer = tokenizer

        if text_embedding is not None:
            self.text_embedding = text_embedding

        self.lm_opt_strategy = lm_opt_strategy

        if self.language_model is not None:
            if lm_opt_strategy == 'lora':
                lora.mark_only_lora_as_trainable(self.language_model)
            
            if self.item_text_field is not None:
                if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
                    lm_proj_in = language_model.config.hidden_size
                    self.lm_projection = nn.Sequential(
                                            MLPModule(
                                                mlp_layers=[lm_proj_in] + config['language_model']['hidden_layers'] + [self.embed_dim],
                                                activation_func=config['language_model']['activation'],
                                                dropout=config['language_model']['dropout']
                                            ),
                                            nn.LayerNorm(self.embed_dim)
                                        )
                else:
                    lm_proj_in0 = language_model.config.word_embed_proj_dim
                    lm_proj_out0 = self.embed_dim * len(self.item_fields)
                    self.lm_projection_of_align = nn.Sequential(
                                                    MLPModule(
                                                        mlp_layers=[lm_proj_in0, lm_proj_out0],
                                                        activation_func=config['language_model']['activation'],
                                                        dropout=config['language_model']['dropout']
                                                    ),
                                                    nn.LayerNorm(lm_proj_out0)
                                                )
                    self.lm_projection = nn.Sequential(
                                            MLPModule(
                                                mlp_layers=[lm_proj_out0, self.embed_dim],
                                                activation_func=config['language_model']['activation'],
                                                dropout=config['language_model']['dropout']
                                            ),
                                            nn.LayerNorm(self.embed_dim)
                                        )
        elif hasattr(config, 'train') and not config['train']['do_train'] and \
            hasattr(config, 'do_infer') and config['do_infer'] and \
            text_embedding is not None:
            lm_proj_in = text_embedding.weight.shape[-1]
            self.lm_projection = nn.Sequential(
                                    MLPModule(
                                        mlp_layers=[lm_proj_in] + config['language_model']['hidden_layers'] + [self.embed_dim],
                                        activation_func=config['language_model']['activation'],
                                        dropout=config['language_model']['dropout']
                                    ),
                                    nn.LayerNorm(self.embed_dim)
                                )
            

        self.embedding = MyEmbeddings(ctr_fields, self.embed_dim, dataset)

        self.lm_state = ModelState.ON
        self.ctr_state = ModelState.ON

        # assert lm_projection_group is ctr_model, cuz we don't adapt CoTrainer to the other setting.
        assert lm_projection_group in ['ctr_model', 'language_model'], f'Expect `lm_projection_group` to be ctr_model or language_model, but got {lm_projection_group}.'
        self.lm_projection_group = lm_projection_group
        

    def init_parameters(self, init_method='xavier_normal'):
        init_methods = {
            'xavier_normal': init.xavier_normal_initialization,
            'xavier_uniform': init.xavier_uniform_initialization,
            'normal': init.normal_initialization(),
        }
        init_method = init_methods[init_method]
        for name, module in self.named_children():
            if 'language_model' not in name and 'text_embedding' not in name:
                module.apply(init_method)
            elif 'text_embedding' in name and module.weight.requires_grad:
                 module.apply(init_method)


    def forward(self, **batch):
        pass


    def _get_item_feat(self, batch):
        if len(self.item_fields) == 1:
            return {self.fiid: batch[self.fiid]}
        elif self.item_text_field is not None and \
            hasattr(self, 'text_embedding') and self.text_embedding is not None:
            ret = dict((field, value) for field, value in batch.items() if field in self.item_fields - {self.item_text_field})
            text_feat = batch[self.fiid]
            return {**ret, self.item_text_field: text_feat}
        else:
            return dict((field, value) for field, value in batch.items() if field in self.item_fields)
        
            
    def _get_user_feat(self, batch):
        if len(self.user_fields) == 1:
            return {self.fuid: batch[self.fuid]}
        else:
            return dict((field, value) for field, value in batch.items() if field in self.user_fields)
        
    def _get_context_feat(self, batch):
        return dict((field, value) for field, value in batch.items() if field in self.context_fields)


    def _get_behavior_feat(self, batch):
        if len(self.item_fields) == 1:
            return {'in_'+self.fiid: batch['in_'+self.fiid]}
        elif self.item_text_field is not None and \
            hasattr(self, 'text_embedding') and self.text_embedding is not None:
            ret = dict((field, value) for field, value in batch.items() if field in self.behavior_fields - {'in_'+self.item_text_field})
            text_feat = batch['in_'+self.fiid]
            return {**ret, 'in_'+self.item_text_field: text_feat}
        else:
            return dict((field, value) for field, value in batch.items() if field in self.behavior_fields)
 
        
    def get_item_embeddings(self, batch):
        item_feat = self._get_item_feat(batch)

        if self.item_text_field is not None:
            item_text_feat = item_feat.pop(self.item_text_field)

            if hasattr(self, 'text_embedding') and self.text_embedding is not None:
                assert ((self.lm_state == ModelState.OFF or self.training == False) and self.text_embedding.embedding_dim != self.embed_dim) or \
                        (self.text_embedding.embedding_dim == self.embed_dim), \
                    'BaseRanker should freeze language_model when equipped with text_embedding and training.'
                item_text_embs = self.text_embedding(item_text_feat)
            else:
                assert not hasattr(self, 'text_embedding') or self.text_embedding is None, \
                    'BaseRanker should not have text_embedding when equipped with language_model.'
                item_text_embs = self.language_model(**item_text_feat, return_dict=True)

                if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
                    item_text_embs = item_text_embs.last_hidden_state[:, 0, :]
                else:
                    item_text_embs = item_text_embs.last_hidden_state

            if self.text_embedding.embedding_dim != self.embed_dim:
                item_text_embs = self.lm_projection(item_text_embs)
                # if self.language_model is not None and \
                #     self.language_model.config.model_type not in ENCODER_TYPE_MODELS:
                #     bs, sequence_lengths = item_text_embs.shape[:2]
                #     sequence_lengths = (torch.eq(
                #                                 item_text_feat['input_ids'], 
                #                                 self.language_model.config.pad_token_id
                #                             ).long().argmax(-1) - 1
                #                         ).to(item_text_embs.device)
                #     item_text_embs = item_text_embs[torch.arange(bs, device=item_text_embs.device), sequence_lengths]

        item_ctr_embs = self.embedding(item_feat)
        if self.item_text_field is not None:
            item_embs = torch.cat([item_ctr_embs, item_text_embs], -1)
        else:
            item_embs = item_ctr_embs
        return item_embs


    def get_user_embeddings(self, batch):
        user_feat = self._get_user_feat(batch)
        user_embs = self.embedding(user_feat)
        return user_embs
    

    def get_context_embeddings(self, batch):
        context_feat  = self._get_context_feat(batch)
        context_embs = self.embedding(context_feat)
        return context_embs
    

    def get_behavior_embeddings(self, batch):
        behavior_feat = self._get_behavior_feat(batch)
        if self.item_text_field is not None:
            behavior_text_feat = behavior_feat.pop('in_'+self.item_text_field)
            if hasattr(self, 'text_embedding') and self.text_embedding is not None:
                assert ((self.lm_state == ModelState.OFF or self.training == False) and self.text_embedding.embedding_dim != self.embed_dim) or \
                        (self.text_embedding.embedding_dim == self.embed_dim), \
                    'BaseRanker should freeze language_model when equipped with non-trainable text_embedding and training.'
                behavior_text_embs = self.text_embedding(behavior_text_feat)
                if self.text_embedding.embedding_dim != self.embed_dim:
                    behavior_text_embs = self.lm_projection(behavior_text_embs)
            else:
                assert not hasattr(self, 'text_embedding') or self.text_embedding is None, \
                    'BaseRanker should not have text_embedding when equipped with language_model.'
                # Cancel padding
                behavior_mask = behavior_text_feat['attention_mask'].sum(-1)                                # bs x behavior_len
                behavior_text_feat = {k : v.flatten(end_dim=-2) for k, v in behavior_text_feat.items()}     # bsxbehavior_len x seq_len 

                non_pad_behavior_text_feat = defaultdict(list)
                nz = behavior_mask.flatten().nonzero()
                non_pad_behavior_text_feat['input_ids'] = behavior_text_feat['input_ids'][nz.flatten()]
                non_pad_behavior_text_feat['attention_mask'] = behavior_text_feat['attention_mask'][nz.flatten()]

                non_pad_behavior_text_embs = self.language_model(**non_pad_behavior_text_feat, return_dict=True).pooler_output
                non_pad_behavior_text_embs = self.lm_projection(non_pad_behavior_text_embs)

                # Padding
                behavior_text_embs = torch.zeros(behavior_mask.flatten().shape[0], non_pad_behavior_text_embs.shape[-1], device=behavior_mask.device)   # bsxbehavior_len x D
                # according to index, scatter src to input, given the dim 
                behavior_text_embs = behavior_text_embs.scatter_(                                                               # input: bsxbehavior_len x D
                                            0,                                                                                  # dim
                                            nz.tile(non_pad_behavior_text_embs.shape[-1]),                                      # index: non-pad-num x D
                                            non_pad_behavior_text_embs)                                                         # src: non-pad-num x D
                behavior_text_embs = behavior_text_embs.reshape(*behavior_mask.shape, -1)                                       # bs x behavior_len x D

        behavior_ctr_embs = self.embedding(behavior_feat)
        if self.item_text_field is not None:
            behavior_embs = torch.cat([behavior_ctr_embs, behavior_text_embs], -1)
        else:
            behavior_embs = behavior_ctr_embs
        return behavior_embs
    

    def get_pooled_behavior_embeddings(self, batch):
        length = (batch['in_'+self.fiid] > 0).float().sum(dim=-1, keepdim=False)
        length = torch.where(length > 0, length, 1)
        behavior_embs = self.get_behavior_embeddings(batch)
        pooled_behavior_embs = self.embedding.seq_pooling_layer(behavior_embs, length)
        return pooled_behavior_embs


    def get_embedddings(self, batch):
        user_emb = self.get_user_embeddings(batch)
        item_emb = self.get_item_embeddings(batch)
        pooled_behavior_emb = self.get_pooled_behavior_embeddings(batch)
        embs = [user_emb, item_emb, pooled_behavior_emb]
        if len(self.context_fields) > 0:
            context_emb = self.get_context_embeddings(batch)
            embs.append(context_emb)
        emb = torch.cat(embs, -1)
        return emb
    

    def from_pretrained(self, pretrained_dir, part='all', in_align=False):
        if not in_align:
            state_dict = torch.load(os.path.join(pretrained_dir, WEIGHTS_NAME))
        else:
            state_dict = torch.load(os.path.join(pretrained_dir, ALIGN_WEIGHTS_NAME))

        if part == 'all':
            if 'text_embedding.weight' in state_dict:
                state_dict.pop('text_embedding.weight')
            if hasattr(self, 'language_model'):
                for _ in set(self.state_dict()) - set(state_dict):                          # self has no pretrained CTR part, and has only language part
                    assert _.startswith('language_model')
            assert len(set(state_dict.keys()) - set(self.state_dict().keys())) == 0     # pretrained one has CTR part, no language part
            self.load_state_dict(state_dict, strict=False)                              # False if first time lora; else True
        else:
            part_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(part):
                    part_state_dict[k.replace(part + '.', '')] = v
            if not in_align:
                getattr(self, part).load_state_dict(part_state_dict, strict=True)
            else:
                getattr(self, part+'_of_align').load_state_dict(part_state_dict, strict=True)


    def freeze(self, model_type):
        if model_type == 'ctr_model':
            self.ctr_state = ModelState.OFF
        elif model_type == 'language_model':
            self.lm_state = ModelState.OFF
        else:
            raise ValueError(f'`model_type` should be ctr_model or language_model, but got {model_type}.')
            
        for n, p in self.named_parameters():
            if 'lm_projection' not in n:
                if ('language_model' in n and model_type == 'language_model') or ('language_model' not in n and model_type == 'ctr_model'):
                    p.requires_grad = False
            else:
                if (model_type == 'language_model' and self.lm_projection_group == 'language_model') or \
                    (model_type == 'ctr_model' and self.lm_projection_group == 'ctr_model') :
                    p.requires_grad = False
                

    def wake(self, model_type):
        if model_type == 'ctr_model':
            self.ctr_state = ModelState.ON
        elif model_type == 'language_model':
            self.lm_state = ModelState.ON
        else:
            raise ValueError(f'`model_type` should be ctr_model or language_model, but got {model_type}.')

        for n, p in self.named_parameters():
            if 'lm_projection' not in n:
                # wake language model
                # not lora strategy: all parts
                # lora startegy: lora parts
                if 'language_model' in n and model_type == 'language_model':
                    p.requires_grad = True
                if model_type == 'language_model' and self.lm_opt_strategy == 'lora':
                    lora.mark_only_lora_as_trainable(self.language_model)   # mark non-lora part as False
                # wake ctr model
                if ('language_model' not in n) and ('text_embedding' not in n) and model_type == 'ctr_model':
                    p.requires_grad = True
            else:
                if (model_type == 'language_model' and self.lm_projection_group == 'language_model') or \
                    (model_type == 'ctr_model' and self.lm_projection_group == 'ctr_model') :
                    p.requires_grad = True


    def switch_states(self):
        if self.ctr_state == ModelState.OFF:
            self.wake('ctr_model')
        else:
            self.freeze('ctr_model')

        if self.lm_state == ModelState.OFF:
            self.wake('language_model')
        else:
            self.freeze('language_model')


    def build_text_embedding(self, batch_size=128, saved_dir=None, device='cuda:0'):
        if hasattr(self, 'text_embedding') and self.text_embedding is not None:
            return

        original_device = self.language_model.device

        if saved_dir is not None:
            path = os.path.join(saved_dir, FROZEN_TEXT_EMBEDDINGS_NAME)
            if os.path.exists(path):
                self.text_embedding = torch.nn.Embedding(
                                        len(self.item_text_feat), 
                                        self.language_model.config.hidden_size,
                                        padding_idx=0
                                    )
                self.text_embedding.load_state_dict(torch.load(path).to(original_device))
                for p in self.text_embedding.parameters():
                    p.requires_grad = False
                return
            else:
                if not os.path.exists(saved_dir):
                    os.makedirs(saved_dir)

        if original_device == torch.device('cpu'):
            self.language_model = self.language_model.to(device)
        else:
            device = original_device

        text_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.item_text_feat), batch_size)):
                text = dict(
                            self.tokenizer.pad(
                                self.item_text_feat[i : i + batch_size],
                                padding=True,
                                return_tensors='pt'
                            )
                        )
                text = {k: v.to(device) for k, v in text.items()}
                text_emb = self.language_model(**text, return_dict=True)
                if self.language_model.config.model_type in ENCODER_TYPE_MODELS:
                    text_emb = text_emb.last_hidden_state[:, 0, :].to('cpu')
                    text_embeddings.append(text_emb)
                else:
                    self.lm_projection_of_align = self.lm_projection_of_align.to(device)
                    text_proj = self.lm_projection_of_align(text_emb.last_hidden_state)
                    sequence_lengths = (torch.eq(
                                            text['input_ids'], 
                                            self.language_model.config.pad_token_id
                                        ).long().argmax(-1) - 1
                                    ).to(device)
                    text_proj = text_proj[torch.arange(text_proj.shape[0], device=text_proj.device), sequence_lengths].to('cpu')   # B x D2
                    text_embeddings.append(text_proj)

            text_embeddings = torch.cat(text_embeddings, dim=0)     # N x D
            text_embedding = nn.Embedding.from_pretrained(
                                text_embeddings,
                                freeze=True,
                                padding_idx=0)
        
        self.text_embedding = text_embedding
        for p in self.text_embedding.parameters():
            p.requires_grad = False

        if saved_dir is not None:
            torch.save(text_embedding, path)

        self.language_model = self.language_model.to(original_device)
        self.text_embedding = self.text_embedding.to(original_device)

