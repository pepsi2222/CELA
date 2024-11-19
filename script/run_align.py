import os
import sys
sys.path.append('.')
sys.path.append('RecStudio')
import pickle
import torch
import numpy as np
from pydantic.utils import deep_update
from utils.utils import get_model
from utils.argument import ModelArguments, DataTrainingArguments
import transformers
from transformers import (
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback, 
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    set_seed)
from transformers.trainer_utils import get_last_checkpoint
from RecStudio.recstudio.utils import parser_yaml, color_dict_normal
from dataset import AlignDataset, Collator
import logging
import datetime
from module.trainer import AlignTrainer
from offline.build_dataset import OfflineDataset

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
                        
def run_align(dataset_dir : str, drop_fields : set = set(), mode : str = 'light', 
              cm_proj=False, **kwargs):
    
    # Model arguments
    align_model_class, align_model_conf = get_model('Align')

    # Arguments    
    conf = deep_update(parser_yaml('config/transformers.yaml'), align_model_conf)

    if mode == 'debug':
        conf['train']['use_cpu'] = True
        conf['train']['dataloader_num_workers'] = 0

    if kwargs is not None:
        conf['train'].update({k: v for k, v in kwargs.items() if k in conf['train']})
        conf['model'].update({k: v for k, v in kwargs.items() if k in conf['model']})
        align_model_conf['loss'].update({k: v for k, v in kwargs.items() if k in align_model_conf['loss']})
        conf.update({k: v for k, v in kwargs.items() if k in conf})

    basename = ''
    if not cm_proj:
        basename += 'wo_cm_proj'
        align_model_conf['ctr_model']['projection_layers'] = None

    if torch.cuda.device_count() == 1:
        basename += f'-bs{conf["train"]["per_device_train_batch_size"]}'
    else:
        basename += f'-perdevicebs{conf["train"]["per_device_train_batch_size"]}'
        if torch.cuda.device_count() > 2:
            raise ValueError('Num of GPUs should not be more than 2.')
        
    if align_model_conf['loss']['loss_fn'] == 'mse':
        basename += '-mse'
    elif align_model_conf['loss']['loss_fn'] == 'bpr':
        basename += f"-bpr-neg{align_model_conf['loss']['neg_count']}"
    elif align_model_conf['loss']['loss_fn'] == 'infonce':
        basename += f"-infonce-temperature{align_model_conf['loss']['temperature']}"
        # if align_model_conf['loss']['simcse'] == True:
        #     basename += '-simcse'
    else:
        raise ValueError("Wrong loss_fn")
    if align_model_conf['loss']['score_fn'] != 'cos':
        basename += f"-{align_model_conf['loss']['score_fn']}"
    basename += f"-lr{conf['ctr_model']['learning_rate']}-wd{conf['ctr_model']['weight_decay']}-dropout{conf['ctr_model']['dropout']}"
    conf['train']['output_dir'] = os.path.join(conf['train']['output_dir'], basename)

    training_args = TrainingArguments(**conf['train'])
    model_args = ModelArguments(**conf['model'])

    set_seed(training_args.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    transformers.utils.logging.add_handler(
        logging.FileHandler(
            os.path.join(training_args.output_dir, f'log.log')))
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
    else:
        raise ValueError(f'`config_name` or `model_name_or_path` should not be None.')

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
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
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
    else:
        logger.info("Training new model from scratch")
        language_model = AutoModel.from_config(config, trust_remote_code=model_args.trust_remote_code)


    # Datasets
    with open(os.path.join(dataset_dir, 'truncated20_dataset.pkl'), 'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        _ = pickle.load(f)
        dataset = pickle.load(f)

    popularity = dataset.inter_feat.groupby(dataset.fiid).count().reindex(np.arange(dataset.num_items), fill_value=0)[dataset.fuid]
    if conf['loss']['weighted_by_item_popularity']:
        # version 1
        # popularity = (dataset.inter_feat.groupby(dataset.fiid).count().reindex(np.arange(dataset.num_items), fill_value=0)[dataset.fuid] + 1) / (dataset.num_inters + 1)
        # version 2
        # dataset.inter_feat.groupby(dataset.fiid).count().reindex(np.arange(dataset.num_items), fill_value=0)[dataset.fuid]
        # version 3
        popularity = np.log10(1 + popularity)
        assert popularity.index.to_list() == dataset.item_feat.index.to_list()

    dataset.dataframe2tensors()
    fields = {dataset.frating}
    if dataset.item_feat is not None:
        dataset.item_feat.del_fields(keep_fields=set(dataset.item_feat.fields) - drop_fields)
        fields = fields.union(dataset.item_feat.fields)

    if dataset.user_feat is not None:
        dataset.user_feat.del_fields(keep_fields=set(dataset.user_feat.fields) - drop_fields)
        fields = fields.union(dataset.user_feat.fields)

    if dataset.name == 'app_gallery':
        fields = fields.union(set(dataset.user_fields + dataset.context_fields))

    fields = sorted(list(fields))

    if conf['loss']['weighted_by_item_popularity']:
        # By no adding popularity field, you can still get all fields when calling __getitem__ from TensorFrame,
        # while keeping item_fields the same.
        # dataset.item_feat.fields.append('popularity')
        dataset.item_feat.data['popularity'] = torch.from_numpy(popularity.to_numpy(np.float32))


    trn_dataset, val_dataset, tst_dataset = AlignDataset(
                                                dataset, 
                                                tokenizer, 
                                                conf['data']['max_seq_length'],
                                                popularity,
                                                conf['tophot_percent']
                                            ).build(align_model_conf['split_ratio'])
    

    # Ctr model
    ctr_fields = sorted(list(set(fields) - {trn_dataset.item_text_field}))
    ctr_model_class, ctr_model_conf = get_model(align_model_conf['ctr_model']['name'])
    ctr_model = ctr_model_class(ctr_model_conf, dataset, ctr_fields)
    if align_model_conf['ctr_model']['pretrained_dir'] is not None:
        ctr_model.from_pretrained(align_model_conf['ctr_model']['pretrained_dir'])
    else:
        raise ValueError('ctr_model/pretrained_dir should not be None.')

    
    if mode != 'tune':
        # Model
        model = align_model_class(
                    config=align_model_conf, 
                    language_model=language_model, 
                    ctr_model=ctr_model, 
                    item_text_field=trn_dataset.item_text_field,
                    ctr_fields=ctr_fields,
                )
        # Callback
        callbacks = [EarlyStoppingCallback(**conf['early_stop'])]
        model.freeze('ctr_model')
        
        # Trainer
        trainer = AlignTrainer(
            model=model,
            args=training_args,
            data_collator=Collator(tokenizer),
            train_dataset=trn_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
        )

        # Training
        # if training_args.resume_from_checkpoint is not None:
        #     checkpoint = training_args.resume_from_checkpoint
        # else:
        #     # Detecting last checkpoint
        #     if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        #         last_checkpoint = get_last_checkpoint(training_args.output_dir)
        #         if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        #             raise ValueError(
        #                 f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        #                 "Use --overwrite_output_dir to overcome.")
        #         elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        #             logger.info(
        #                 f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
        #                 "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        #     checkpoint = last_checkpoint
            
        # trn_result = trainer.train(resume_from_checkpoint=checkpoint)
        trn_result = trainer.train()
        trn_metrics = trn_result.metrics
        trn_metrics["train_samples"] = len(trn_dataset)
        trn_metrics["validate_samples"] = len(val_dataset)

        trainer.log_metrics("train", trn_metrics)
        trainer.save_metrics("train", trn_metrics)
        trainer.save_state()
        # model.save(training_conf['output_dir'])


        # Testing
        # if training_args.do_eval:
        #     logger.info("*** Evaluate ***")
        #     tst_metrics = trainer.evaluate(tst_dataset)
        #     tst_metrics["test_samples"] = len(tst_dataset)
        #     trainer.log_metrics("test", tst_metrics)
        #     trainer.save_metrics("test", tst_metrics)


    elif mode == 'tune':
        pass


if __name__ == '__main__':
    dataset = 'amazon-toys'
    if dataset == 'amazon-toys':
        dataset_dir = 'Toys_and_Games'
    elif dataset == 'amazon-sports':
        dataset_dir = 'Sports_and_Outdoors'
    elif dataset == 'movielens':
        dataset_dir = 'MovieLens'
    elif dataset == 'steam':
        dataset_dir = 'Steam'

    print(sys.argv[1:])

    model_name_or_path = os.path.join(os.getenv('DATA_MOUNT_DIR'), 'simcse_mlm', sys.argv[1])
    for i in os.listdir(model_name_or_path):
        if 'checkpoint-' in i:
            model_name_or_path = os.path.join(model_name_or_path, i)
    if 'checkpoint-' not in model_name_or_path:
        raise ValueError('Please load the best align checkpoint.')
    
    run_align(
        dataset_dir=os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_dir),
        mode='light',
        model_name_or_path=model_name_or_path,
        # output_dir=os.path.join('align', sys.argv[1])
    )