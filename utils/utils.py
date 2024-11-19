import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame, Series
import importlib
from functools import partial
from pydantic.utils import deep_update
from RecStudio.recstudio.utils import parser_yaml
from RecStudio.recstudio.data.dataset import TensorFrame
from datasets import Dataset as TFDataset
from transformers.utils import logging, ExplicitEnum

FROZEN_TEXT_EMBEDDINGS_NAME = 'text_embeddings.pt'

class ModelState(ExplicitEnum):
    ON = 'on'
    OFF = 'off'


def get_model(model_name: str):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        torch.nn.Module: model class
        Dict: model configuration dict
    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)

    fname = os.path.join('config', 'baseranker.yaml')
    conf = parser_yaml(fname)

    fname = os.path.join('config', model_file_name+'.yaml')
    if os.path.isfile(fname):
        conf = deep_update(conf, parser_yaml(fname))
    return model_class, conf


def build_text_embedding(language_model, tokenizer, item_text_feat: Series, max_seq_len, save_dir, batch_size=128, device='cuda:1'):
    language_model = language_model.to(device)
    path = os.path.join(save_dir, FROZEN_TEXT_EMBEDDINGS_NAME)
    if os.path.exists(path):
        text_embedding = torch.load(path)
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        f = item_text_feat.name
        item_text_feat = tokenize_text_fields(item_text_feat, tokenizer, f, max_seq_len)
        text_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(item_text_feat), batch_size)):
                text = dict(
                            tokenizer.pad(
                                item_text_feat[i : i + batch_size],
                                padding=True,
                                return_tensors='pt'
                            )
                        )
                text = {k: v.to(device) for k, v in text.items()}
                text_embeddings.append(language_model(**text, return_dict=True).last_hidden_state[:, 0, :].to('cpu'))
            text_embeddings = torch.cat(text_embeddings, dim=0)
            text_embedding = nn.Embedding.from_pretrained(
                                text_embeddings,
                                freeze=True,
                                padding_idx=0)
        torch.save(text_embedding, path)

    return text_embedding
    

    
def tokenize_text_fields(item_text_feat : Series, tokenizer, item_text_field, max_seq_len):
    f = item_text_field
    item_text_feat = pd.DataFrame(item_text_feat)
    item_text_feat[f][item_text_feat[f] == 0] = tokenizer.unk_token
    item_text_feat = TFDataset.from_pandas(item_text_feat, preserve_index=False)

    def tokenize_a_serie(df, field, tokenizer, max_seq_len):
        return tokenizer(
                    df[field], 
                    add_special_tokens=False, 
                    truncation=True,
                    max_length=max_seq_len, 
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
    
    item_text_feat = item_text_feat.map(
                        partial(
                            tokenize_a_serie,
                            field=f,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len
                        ),
                        remove_columns=[f],
                        batched=True
                    )
    return item_text_feat
    

def get_item_text_field(dataset : TensorFrame):
    item_text_fields = []
    for f, t in dataset.field2type.items():
        if t == 'text' and f in dataset.item_feat.fields:
            item_text_fields.append(f)

    if len(item_text_fields) != 1:
        raise ValueError(f'item_text_fields is {item_text_fields}, which should be a length of 1.')
    return item_text_fields[0]
