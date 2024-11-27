import os
import sys
sys.path.append('.')
sys.path.append('RecStudio')
import pickle
import torch
import copy
import numpy as np
import pandas as pd
from pydantic.utils import deep_update
from utils.utils import get_model, get_item_text_field, tokenize_text_fields, build_text_embedding
from utils.argument import ModelArguments
import transformers
from transformers import (
    TrainingArguments, 
    EarlyStoppingCallback, 
    AutoTokenizer,
    AutoConfig,
    AutoModel, 
    set_seed)
from RecStudio.recstudio.utils import parser_yaml, color_dict_normal
from dataset import CTRDataset, CTRDatasetWithText, Collator
import logging
from utils import ENCODER_TYPE_MODELS, DECODER_TYPE_MODELS
from utils.metrics import compute_metrics_for_ctr
from module.trainer import CoTrainer, IntervalAlterTrainer
from module.callback import AlterCallback, ProcessTextEmbeddingCallback
from offline.build_dataset import OfflineDataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
                        
def run_cotrain(model : str, dataset_dir : str, model_config_path : str = None, mode : str = 'light', 
                train_lm : str = 'none',                    # none / always / interval-alter / only
                lm_opt_strategy : str = 'lora',             # none / lora / all
                lm_projection_group : str = 'ctr_model',    # ctr_model / language_model
                **kwargs):
    
    # Datasets
    with open(os.path.join(dataset_dir, 'truncated20_dataset.pkl'), 'rb') as f:
        trn_data = pickle.load(f)
        val_data = pickle.load(f)
        tst_data = pickle.load(f)
        dataset = pickle.load(f)

    # Model arguments
    model_class, model_conf = get_model(model)
    if model_config_path is not None:
        model_conf = deep_update(model_conf, parser_yaml(model_config_path))

    # Arguments
    conf =  deep_update(parser_yaml('config/transformers.yaml'), parser_yaml('config/cotrain.yaml'))
    conf = deep_update(conf, model_conf)
    conf['train']['label_names'] = [dataset.frating]

    if mode == 'debug':
        conf['train']['use_cpu'] = True
        conf['train']['dataloader_num_workers'] = 0
        conf['train']['fp16'] = False

    if kwargs is not None:
        conf['train'].update({k: v for k, v in kwargs.items() if k in conf['train']})
        conf['model'].update({k: v for k, v in kwargs.items() if k in conf['model']})
        conf['ctr_model'].update({k.replace('ctr_model_', ''): v for k, v in kwargs.items() if k.replace('ctr_model_', '') in conf['ctr_model']})
        conf.update({k: v for k, v in kwargs.items() if k in conf}) # pretrained_dir
        

    if conf['pretrained_dir'] is not None and train_lm in ['always', 'only', 'interval-alter']:
        assert train_lm != 'none', f'Expect train_lm to be `always`, `interval-alter`, `only` given pretrained_dir, but got {train_lm}.'
        conf['train']['per_device_train_batch_size'] = 8
        if train_lm in ['always', 'only']:
            conf['train']['evaluation_strategy'] = conf['train']['logging_strategy'] = conf['train']['save_strategy'] = 'steps'
            conf['early_stop']['early_stopping_patience'] = 50
            print('Set per device batch size to be 8, eval/log/save stategy to be steps, early stop patience to be 50.')


    ori_output_dir = conf['train']['output_dir']
    assert torch.cuda.device_count() == 2, 'Num of GPUs should be 2 when cotraining.'
    if train_lm == 'none':
        basename = f'ctr_lr{conf["ctr_model"]["learning_rate"]}_wd{conf["ctr_model"]["weight_decay"]}_dropout{conf["ctr_model"]["dropout"]}_perdevicebs{conf["train"]["per_device_train_batch_size"]}'
    else:
        basename = f'ctr_lr{conf["ctr_model"]["learning_rate"]}_wd{conf["ctr_model"]["weight_decay"]}_dropout{conf["ctr_model"]["dropout"]}_perdevicebs{conf["train"]["per_device_train_batch_size"]}_lm_lr{conf["train"]["learning_rate"]}'
    
    # if conf['pretrained_dir'] is not None:
    #     pretrained_basename = os.path.basename(os.path.basename(conf['pretrained_dir'].strip('/')))
    #     if pretrained_basename == basename:
    #         basename = '1-' + basename + f'-2-{train_lm}-{lm_opt_strategy}'
    #     else:
    #         basename = '1-' + pretrained_basename + f'-2-{basename}-{train_lm}-{lm_opt_strategy}'
        
    #     # if 'alter' in train_lm:
    #     #     basename += f'-lm_proj_in_{lm_projection_group}'
    # elif train_lm != 'none':
    #     basename += f'-{train_lm}'
    if 'gradient_accumulation_steps' in kwargs:
        basename += f'_gradaccstep{conf["train"]["gradient_accumulation_steps"]}'
    conf['train']['output_dir'] = os.path.join(conf['train']['output_dir'], basename)

    

    training_args = TrainingArguments(**conf['train'])
    model_args = ModelArguments(**conf['model'])
    training_args.ctr_train_batch_size = conf['alter_callback']['ctr_per_device_train_batch_size']

    set_seed(training_args.seed)


    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout), 
        ],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    transformers.utils.logging.add_handler(
        logging.FileHandler(os.path.join(training_args.output_dir, f'log.log')))
    transformers.utils.logging.enable_explicit_format()
    logger = transformers.utils.logging.get_logger()
    logger.setLevel(log_level)

    logger.info('****All Configurations****')
    logger.info(color_dict_normal(conf))


    # Config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # Lora
    if lm_opt_strategy == 'lora' and conf['pretrained_dir'] is not None:
        config.apply_lora = True
        config.lora_r = conf['lora']['r']
        config.lora_alpha = conf['lora']['alpha']


    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        tokenizer = None
    

    # Language model
    if model_args.model_name_or_path:
        language_model = AutoModel.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            token=model_args.token,
                            trust_remote_code=model_args.trust_remote_code,
                            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                        )
        text_embedding = None
    else:
        logger.info("No model_name_or_path, so initialize a random embedding table.")
        language_model = None
        if conf['do_infer']:
            text_embedding = torch.nn.Embedding(dataset.num_items, config.hidden_size, padding_idx=0)
        else:
            text_embedding = torch.nn.Embedding(dataset.num_items, model_conf['ctr_model']['embed_dim'], padding_idx=0)

    # Popularity
    if dataset.user_feat is not None:
        popularity_u = dataset.inter_feat.groupby(dataset.fuid).count().reindex(np.arange(dataset.num_users), fill_value=0)[dataset.fiid]
    else:
        popularity_u = None
    popularity_i = dataset.inter_feat.groupby(dataset.fiid).count().reindex(np.arange(dataset.num_items), fill_value=0)[dataset.fuid]

    # Datasets
    dataset.dataframe2tensors()
    item_text_field = get_item_text_field(dataset)
    if tokenizer is not None:
        item_text_feat = tokenize_text_fields(
                            dataset.item_feat.get_col(item_text_field), 
                            tokenizer,
                            item_text_field,
                            conf['data']['max_seq_length']
                        )
    else:
        item_text_feat = None
    
    if train_lm != 'none':
        trn_dataset = CTRDatasetWithText(
                        dataset, trn_data, tokenizer, conf['data']['max_seq_length'])
        trn_dataset.tokenize_text_fields()
        item_text_field = trn_dataset.item_text_field

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.item_feat.del_fields(keep_fields=set(eval_dataset.item_feat.fields) - {item_text_field})
    else:
        dataset.item_feat.del_fields(keep_fields=set(dataset.item_feat.fields) - {item_text_field})
        trn_dataset = CTRDataset(dataset, trn_data)
        eval_dataset = dataset
    tst_dataset = CTRDataset(eval_dataset, tst_data)
    val_dataset = CTRDataset(eval_dataset, val_data) if len(val_data) != 0 else tst_dataset
        
    if dataset.name != 'app_gallery':
        ctr_fields = {dataset.fuid, dataset.fiid, dataset.frating}.union(
                    dataset.item_feat.fields).union(dataset.user_feat.fields) - \
                    {item_text_field}
    else:
        ctr_fields = set(dataset.user_fields + dataset.context_fields + dataset.item_fields + [dataset.frating]) - {item_text_field}
    
    ctr_fields = sorted(list(ctr_fields))

    
    if mode != 'tune':
        # Callback
        callbacks = [EarlyStoppingCallback(**conf['early_stop'])]

        # Model
        model = model_class(
                    config=conf, 
                    dataset=dataset, 
                    ctr_fields=ctr_fields, 
                    item_text_field=item_text_field, 
                    language_model=language_model,
                    item_text_feat=item_text_feat,
                    tokenizer=tokenizer,
                    lm_opt_strategy=lm_opt_strategy,
                    lm_projection_group=lm_projection_group,
                    text_embedding=text_embedding,
                )           
        
        if conf['pretrained_dir_for_lm_part'] is not None:
            logger.info(f'You are training a model with {lm_opt_strategy} params in LM, which is loaded from {conf["pretrained_dir_for_lm_part"]}.')
            logger.info('You should load a pretrained model, so you should change two learning_rates.')
            model.from_pretrained(pretrained_dir=conf["pretrained_dir_for_lm_part"], part='language_model')

        if conf['pretrained_dir'] is not None:
            logger.info(f'You are training a model with {lm_opt_strategy} params in LM, which is loaded from {conf["pretrained_dir"]}.')
            logger.info('You should load a pretrained model, so you should change two learning_rates.')
            model.from_pretrained(
                pretrained_dir=conf["pretrained_dir"], 
                part=conf['load_pretrained_part'],
                in_align=language_model.config.model_type in DECODER_TYPE_MODELS if language_model is not None else False,
            )
        
        if train_lm == 'none':
            if model_args.model_name_or_path:
                model.freeze('language_model')
                model.build_text_embedding()
                model.language_model = None
                del language_model
            trainer_class = CoTrainer

        elif train_lm == 'interval-alter':
            callbacks += [
                            AlterCallback(**conf['alter_callback']),
                            ProcessTextEmbeddingCallback(),
                        ]
            logger.info(f'Using AlterCallback, lm_projection belongs to {lm_projection_group}.')
            trainer_class = IntervalAlterTrainer

        elif train_lm == 'only':
            callbacks += [ProcessTextEmbeddingCallback()]
            model.freeze('ctr_model')
            trainer_class = CoTrainer

        else:
            assert train_lm == 'always', f'train_lm should be in [`none`, `interval-alter`, `always`, `only`], but got {train_lm}.'
            callbacks += [ProcessTextEmbeddingCallback()]
            trainer_class = CoTrainer

        # Trainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            data_collator=Collator(tokenizer),
            train_dataset=trn_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_for_ctr,
            callbacks=callbacks
        )

        # Training
        if training_args.do_train:
            trn_result = trainer.train()
            trn_metrics = trn_result.metrics
            trn_metrics["train_samples"] = len(trn_dataset)
            trn_metrics["validate_samples"] = len(val_dataset)

            trainer.log_metrics("train", trn_metrics)
            trainer.save_metrics("train", trn_metrics)
            trainer.save_state()


        # Infer
        if conf['do_infer']:
            tst_dataset = tst_dataset.build_filtered_dataset(
                                popularity_u,
                                popularity_i,
                                conf['user_bins'], 
                                conf['item_bins'],
                            )

            for d in tst_dataset:
                name = d.name
                tst_metrics = trainer.evaluate(d)
                tst_metrics["test_samples"] = len(d)
                trainer.log_metrics(f"test_{name}", tst_metrics)
                trainer.save_metrics(f"test_{name}", tst_metrics)
            


    elif mode == 'tune':
        pass


if __name__ == '__main__':
    dataset = sys.argv[1]
    if dataset == 'amazon-toys':
        dataset_dir = 'Toys_and_Games'
    elif dataset == 'amazon-sports':
        dataset_dir = 'Sports_and_Outdoors'
    elif dataset == 'movielens':
        dataset_dir = 'MovieLens'
    elif dataset == 'steam':
        dataset_dir = 'Steam'
    

    run_cotrain(
        model='DIN',
        dataset_dir=os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_dir),
        mode='light',
        train_lm='none',
        output_dir=sys.argv[2],
        model_name_or_path=sys.argv[3],
        config_name=os.path.join(os.getenv('DATA_MOUNT_DIR'), sys.argv[4]),
        tokenizer_name=os.path.join(os.getenv('DATA_MOUNT_DIR'), sys.argv[4]),
    )
