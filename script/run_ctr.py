import os
import sys
sys.path.append('.')
sys.path.append('RecStudio')
import pickle
import torch
import copy
import numpy as np
from pydantic.utils import deep_update
from utils.utils import get_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
from RecStudio.recstudio.utils import parser_yaml, color_dict_normal
from dataset import CTRDataset, Collator
from utils.metrics import compute_metrics_for_ctr
import logging
import transformers
from offline.build_dataset import OfflineDataset


def run_ctr(model : str, dataset_dir : str, model_config_path : str = None, drop_fields : set = set(), mode : str = 'light', **kwargs):

    # Datasets
    with open(os.path.join(dataset_dir, 'truncated20_dataset.pkl'), 'rb') as f:
        trn_data = pickle.load(f)
        val_data = pickle.load(f)
        tst_data = pickle.load(f)
        dataset = pickle.load(f)

    # Popularity
    if dataset.user_feat is not None:
        popularity_u = dataset.inter_feat.groupby(dataset.fuid).count().reindex(np.arange(dataset.num_users), fill_value=0)[dataset.fiid]
    else:
        popularity_u = None
    popularity_i = dataset.inter_feat.groupby(dataset.fiid).count().reindex(np.arange(dataset.num_items), fill_value=0)[dataset.fuid]

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

    trn_dataset = CTRDataset(dataset, trn_data)
    tst_dataset = copy.deepcopy(trn_dataset)
    tst_dataset.data = tst_data
    if len(val_data) != 0:
        val_dataset = copy.deepcopy(trn_dataset)
        val_dataset.data = val_data
    else:
        val_dataset = tst_dataset

    # Model arguments
    model_class, model_conf = get_model(model)
    if model_config_path is not None:
        model_conf = deep_update(model_conf, parser_yaml(model_config_path))


    # Training arguments
    conf = deep_update(parser_yaml('config/transformers.yaml'), parser_yaml('config/ctr.yaml'))
    conf = deep_update(conf, model_conf)
    conf['train']['label_names'] = [dataset.frating]
    if mode == 'debug':
        conf['train']['use_cpu'] = True
        conf['train']['dataloader_num_workers'] = 0
        conf['train']['fp16'] = False

    if kwargs is not None:
        conf['train'].update({k: v for k, v in kwargs.items() if k in conf['train']})
        conf['ctr_model'].update({k.replace('ctr_model_', ''): v for k, v in kwargs.items() if k.replace('ctr_model_', '') in conf['ctr_model']})
        conf.update({k: v for k, v in kwargs.items() if k in conf})

    conf['train']['learning_rate'] = model_conf['ctr_model']['learning_rate']
    conf['train']['weight_decay'] = model_conf['ctr_model']['weight_decay']
    conf['train']['lr_scheduler_type'] = model_conf['ctr_model']['scheduler']   # linear / cosine / cosine_with_restarts / polynomial / constant / constant_with_warmup / inverse_sqrt / reduce_lr_on_plateau

    
    basename = f'lr{conf["train"]["learning_rate"]}_bs{conf["train"]["per_device_train_batch_size"]}_wd{conf["train"]["weight_decay"]}_dropout{conf["ctr_model"]["dropout"]}'
    if "mlp_layer" in conf["ctr_model"]:
        basename += f'_mlp{conf["ctr_model"]["mlp_layer"]}'
    if conf['ctr_model']['batch_norm']:
        basename += '_bn'
    conf['train']['output_dir'] = os.path.join(conf['train']['output_dir'], model, basename)

    training_args = TrainingArguments(**conf['train'])

    set_seed(training_args.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
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


    if mode in ['light', 'debug']:
        # Model
        model = model_class(conf, dataset, fields)
        if conf['pretrained_dir'] is not None:
            logger.info(f"Loaded from {conf['pretrained_dir']}.")
            model.from_pretrained(conf['pretrained_dir'])

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=Collator(),
            train_dataset=trn_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_for_ctr,
            callbacks=[EarlyStoppingCallback(**conf['early_stop'])]
        )

        # Training
        if training_args.do_train:
            assert torch.cuda.device_count() == 2, 'Num of GPUs should be 2 when training naive CTR model.'

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
        drop_fields = {'description'}
    
    if dataset == 'amazon-sports':
        dataset_dir = 'Sports_and_Games'
        drop_fields = {'description'}
    
    elif dataset == 'steam':
        dataset_dir = 'Steam'
        drop_fields = {'short_description', 'price', 'release_date', 'supported_languages', 'genres', 'categories', 'mac', 'windows'}
    
    elif dataset == 'movielens':
        dataset_dir = 'MovieLens'
        drop_fields = {'summary'}
    

    run_ctr(
        model='DIN', 
        dataset_dir=os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_dir),
        drop_fields=drop_fields,
        mode='light',
    )
