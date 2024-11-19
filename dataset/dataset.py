import pandas as pd
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch import randperm
from collections import defaultdict
from multiprocessing import Pool
from datasets import Dataset as TFDataset
from functools import partial
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch


def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class CTRDataset(Dataset):
    def __init__(self, dataset, data):
        self.dataset = dataset
        self.data = data
        self.name = 'all'

        self.fuid = dataset.fuid
        self.fiid = dataset.fiid
        self.frating = dataset.frating

        self.item_feat = dataset.item_feat
        self.user_feat = dataset.user_feat                        

            
    def build_filtered_dataset(self, popularity_u, popularity_i, user_bins, item_bins):
        assert self.dataset.name != 'app_gallery'
        if isinstance(user_bins, int):
            user_bins =  np.linspace(self.dataset.config['min_user_inter'], popularity_u.max(), user_bins + 1)  # user_bins + 1
        if isinstance(item_bins, int):
            item_bins = np.linspace(self.dataset.config['min_item_inter'], popularity_i.max(), item_bins + 1)   # item_bins + 1

        filtered_data = defaultdict(list)
        for uid, u_bh, iid, y in self.data:
            for u in range(len(user_bins) - 1):
                for i in range(len(item_bins) - 1):
                    if len(user_bins) == 2:
                        if popularity_i[iid] >  item_bins[i] and popularity_i[iid] <= item_bins[i + 1]:
                            filtered_data[f'ibin({item_bins[i]},{item_bins[i + 1]}]'].append((uid, u_bh, iid, y))
                    elif len(item_bins) == 2:
                        if popularity_u[uid] >  user_bins[u] and popularity_u[uid] <= user_bins[u + 1]:
                            filtered_data[f'ubin({user_bins[u]},{user_bins[u + 1]}]'].append((uid, u_bh, iid, y))
                    else:
                        if popularity_u[uid] >  user_bins[u] and popularity_u[uid] <= user_bins[u + 1] and \
                            popularity_i[iid] >  item_bins[i] and popularity_i[iid] <= item_bins[i + 1]:
                            filtered_data[f'ubin({user_bins[u]},{user_bins[u + 1]}]_ibin({item_bins[i]},{item_bins[i + 1]}]'].append((uid, u_bh, iid, y))
        
        return [self._copy(k, v) for k, v in filtered_data.items()]
                
    def _copy(self, name, data):
        d = copy.copy(self)
        d.data = data
        d.name = name
        return d

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        if self.dataset.name != 'app_gallery':
            uid, u_bh, iid, y = data
            ret = {self.frating: torch.tensor(y)}
            if self.user_feat is not None:
                ret = {**ret, **self.user_feat[uid]}
            if self.item_feat is not None:
                ret = {**ret, **self.item_feat[iid]}

            user_behavior_feat = self.item_feat[u_bh]
            for field, value in user_behavior_feat.items():
                ret['in_'+field] = value

        else:
            u_f, u_bh, iid, y, c_f = data
            ret = {self.frating: torch.tensor(y)}
            if self.item_feat is not None:
                ret = {**ret, **self.item_feat[iid]}

            for f, u in zip(self.dataset.user_fields, u_f):
                ret[f] = torch.tensor(u)
            for f, c in zip(self.dataset.context_fields, c_f):
                ret[f] = torch.tensor(c)

            ret['in_'+self.fiid] = torch.tensor(u_bh)

        return ret
    

class AlignDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, popularity=None, tophot_percent=1.0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.item_feat = dataset.item_feat

        self.tophot_percent = tophot_percent

        self.tokenize_text_fields()
        if tophot_percent == 1.0:
            self.data_index = torch.arange(0, len(self.item_feat))
        else:
            rank = popularity.rank(
                                method='first',
                                ascending=False
                            )
            self.data_index =  torch.tensor(
                                    popularity[
                                        rank <= (tophot_percent * len(self.item_feat))
                                    ].index.to_list())


    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        idx = self.data_index[index]
        ret = self.item_feat[idx]
        return ret

    def build(self, split_ratio):
        lens = [int(len(self) * _) for _ in split_ratio]
        lens[0] = len(self) - sum(lens[1:])
        splits = []
        indices = randperm(len(self))
        for offset, length in zip(_accumulate(lens), lens):
            splits.append(indices[offset - length : offset])
        return [self._copy(_) for _ in splits]
    
    def _copy(self, data_index):
        d = copy.copy(self)
        d.data_index = data_index
        return d
    
    def _get_item_text_feat(self):
        item_text_fields = []
        for f, t in self.dataset.field2type.items():
            if t == 'text' and f in self.dataset.item_feat.fields:
                item_text_fields.append(f)

        if len(item_text_fields) != 1:
            raise ValueError(f'item_text_fields is {item_text_fields}, which should be a length of 1.')
        f = self.item_text_field = item_text_fields[0]

        item_text_feat = pd.DataFrame(self.dataset.item_feat.get_col(f))
        item_text_feat[f][item_text_feat[f] == 0] = self.tokenizer.unk_token
        item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)
        return item_text_feat
    
    def tokenize_text_fields(self):
        item_text_feat = self._get_item_text_feat()

        def tokenize_a_serie(df, field, tokenizer, max_seq_len):
            return tokenizer(
                        df[field], 
                        add_special_tokens=True, 
                        truncation=True,
                        max_length=max_seq_len, 
                        return_attention_mask=False,
                        return_token_type_ids=False
                    )
        
        f = self.item_text_field
        item_text_feat = item_text_feat.map(
                            partial(
                                tokenize_a_serie,
                                field=f,
                                tokenizer=self.tokenizer,
                                max_seq_len=self.max_seq_len
                            ),
                            remove_columns=[f],
                            batched=True
                        )

        self.dataset.item_feat.data[f] = item_text_feat['input_ids']


class CTRDatasetWithText(CTRDataset, AlignDataset):
    def __init__(self, dataset, data, tokenizer=None, max_seq_len=None):
        CTRDataset.__init__(self, dataset, data)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return CTRDataset.__len__(self)
    
    def __getitem__(self, index):
        data = self.data[index]
        if self.dataset.name != 'app_gallery':
            uid, u_bh, iid, y = data
            ret = {self.frating: torch.tensor(y)}
            if self.user_feat is not None:
                ret = {**ret, **self.user_feat[uid]}
            if self.item_feat is not None:
                ret = {**ret, **self.item_feat[iid]}

            user_behavior_feat = defaultdict(list)
            for bh in u_bh:
                for field, value in self.item_feat[bh].items():
                    user_behavior_feat[field].append(value)

            for field, value in user_behavior_feat.items():
                if isinstance(value[0], list):
                    ret['in_'+field] = value
                elif value[0].dim() == 0:
                    ret['in_'+field] = torch.tensor(value)
                else:
                    ret['in_'+field] = pad_sequence(value, batch_first=True, padding_value=0)
        else:
            u_f, u_bh, iid, y, c_f = data
            ret = {self.frating: torch.tensor(y)}
            if self.item_feat is not None:
                ret = {**ret, **self.item_feat[iid]}

            for f, u in zip(self.dataset.user_fields, u_f):
                ret[f] = u
            for f, c in zip(self.dataset.context_fields, c_f):
                ret[f] = c

            ret['in_'+self.fiid] = u_bh
            ret['in_'+self.item_text_field] = [self.item_feat[bh] for bh in u_bh]

        return ret


class Collator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        d = defaultdict(list)
        for _ in batch:
            for field, value in _.items():
                d[field].append(value)
        
        ret = {}
        for field, value in d.items():
            if isinstance(value[0], list):
                if isinstance(value[0][0], list):
                    # assert False, 'waiting to be checked.'
                    tmp = defaultdict(list)
                    max_behavior_len = 0
                    max_seq_len = 0
                    for u_bh_text in value:
                        for k, v in dict(
                                        self.tokenizer.pad(
                                                {'input_ids': u_bh_text},
                                                padding=True,
                                                return_tensors='pt'
                                            )
                                        ).items():
                            max_behavior_len = max(max_behavior_len, v.shape[0])
                            max_seq_len = max(max_seq_len, v.shape[1])
                            tmp[k].append(v)     
                            # tmp: {
                            #           'input_ids': list with a length of bs, each element is a Tensor with shape behavior_len x seq_len
                            #           'attention_mask': list with a length of bs, each element is a Tensor with shape behavior_len x seq_le
                            # }
                    for k, v in tmp.items():
                        for i, _ in enumerate(v):
                            behavior_len, seq_len = _.shape
                            tmp[k][i] = F.pad(_, 
                                              [0, max_seq_len - seq_len, 
                                               0, max_behavior_len - behavior_len])
                        tmp[k] = pad_sequence(v, batch_first=True, padding_value=0)

                    ret[field] = tmp 
                else:
                    ret[field] = dict(
                                    self.tokenizer.pad(
                                        {'input_ids': value},
                                        padding=True,
                                        return_tensors='pt'
                                    )
                                )
            elif value[0].dim() == 0:
                ret[field] = torch.tensor(value)
            else:
                ret[field] = pad_sequence(value, batch_first=True, padding_value=0)
        return ret



class DataCollatorForLanguageModeling4SimCSE(DataCollatorForLanguageModeling):

    def __init__(self, simcse, **kwargs):
        super().__init__(**kwargs)
        self.simcse = simcse

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        
        if self.simcse:
            batch["original_input_ids"] = batch["input_ids"].clone()

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch
    

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise ValueError('You should arouse torch_call.')
    
    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise ValueError('You should arouse torch_call.')