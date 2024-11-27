import os
import re
import sys
work_dir = re.search('(.*LM4REC).*', os.getcwd(), re.IGNORECASE).group(1)
sys.path.append(work_dir)
sys.path.append(os.path.join(work_dir, 'RecStudio'))
import json
import pickle
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from utils import get_data_from_json_by_line
from RecStudio.recstudio.data import SeqDataset
from RecStudio.recstudio.utils import parser_yaml, seed_everything

def extract_item_description(dataset, dataset_dir):
    if 'amazon' in dataset:
        description_field = 'description'
        item_fields = ['asin', 'price', 'category', 'description', 'brand']
    elif dataset == 'steam':
        description_field = 'short_description'
        item_fields = ['product_id', 'short_description', 'price', 'release_date', 'supported_languages', 'genres', 'categories', 'mac', 'windows']
    elif dataset == 'movielens':
        description_field = 'summary'
        item_fields = ['movieId', 'genres', 'summary']
    else:
        raise ValueError(f'Expect `amazon-*`, `steam` or `movielens`, but got {dataset}.')
    
    item_df = pd.read_csv(
                os.path.join(dataset_dir, 'meta_item.csv'),
                sep='\t',
                header=0,
                names=item_fields
            )       
    desc_path = os.path.join(dataset_dir, 'item_desc.txt')
    with open(desc_path, 'w') as wf:
        for desc in item_df[description_field]:
            if desc != '' and not pd.isna(desc):
                wf.write(desc + '\n')

def adapt2recstudio(dataset, dataset_dir):
    if 'amazon' in dataset:
        item_fields = ['asin', 'price', 'category', 'description', 'brand']
        if 'toys' in dataset:
            item_df = pd.read_json(
                        os.path.join(dataset_dir, 'meta_'+os.path.basename(dataset_dir)+'.json'),
                        lines=True
                    )[item_fields]
        else:
            item_df = get_data_from_json_by_line(
                        json_file_path=os.path.join(dataset_dir, 'meta_'+os.path.basename(dataset_dir)+'.json'),
                        fields=item_fields
                    )


        # convert price to float
        price = item_df['price'].map(lambda x : x.replace(',', '') if x.startswith('$') else '$0')
        price = price.map(
            lambda x : float(x.lstrip('$')) if '-' not in x 
            else np.mean(
                [float(_.lstrip('$')) for _ in x.split(' - ')]
                )
            )
        item_df['price'] = price

        # filter category whose number is greater than 5; pad if no category; join categories with sep
        all_categories = []
        for c in item_df['category']:
            all_categories += c
        num_cat = {}
        for c in all_categories:
            if c not in num_cat:
                num_cat[c] = 1
            else:
                num_cat[c] += 1
        category = item_df['category'].map(lambda x: [_ for _ in x if num_cat[_] > 5])  
        category = category.map(lambda x: x if len(x) > 0 else ['[PAD]'])  
        item_df['category'] = category.map(lambda x : '|||'.join(x))

        # process description
        item_df['description'] = item_df['description'].map(lambda x: ' '.join(x).replace('\t', ' ').replace('\n', '. ').replace('\r', '. ')) 
        
        # process brand
        item_df['brand'] = item_df['brand'].map(lambda x: x.lower())

        # write item information to csv
        item_df.to_csv(
            os.path.join(dataset_dir, 'meta_item.csv'),
            sep='\t',
            header=True,
            index=False,
            columns=item_fields
        )

        # write ineractions to csv
        inter_fields = ['asin', 'reviewerID', 'unixReviewTime']
        if 'toys' in dataset:
            inter_df = pd.read_json(
                        os.path.join(dataset_dir, 'Toys_and_Games_5.json'),
                        lines=True
                    )[inter_fields]
            
        elif 'sports' in dataset:
            inter_df = pd.read_csv(
                        os.path.join(dataset_dir, f'{os.path.basename(dataset_dir)}.csv'),
                        sep=',',
                        header=None,
                        names=['asin', 'reviewerID', 'overall', 'unixReviewTime']
                    )[inter_fields]
            
        elif 'books' in dataset:
            inter_df = pd.read_csv(
                        os.path.join(dataset_dir, 'sampled_Books.csv'),
                        sep='\t',
                        header=0
                    )[inter_fields]
            
        elif 'clothing' in dataset:
            inter_df = pd.read_csv(
                        os.path.join(dataset_dir, 'sampled_Clothing_Shoes_and_Jewelry.csv'),
                        sep='\t',
                        header=0
                    )[inter_fields]
        
        inter_df.to_csv(
            os.path.join(dataset_dir, 'processed_'+os.path.basename(dataset_dir)+'.csv'),
            sep='\t',
            header=True,
            index=False,
            columns=inter_fields
        )

    elif dataset == 'movielens':
        item_fields = ['movieId', 'genres', 'summary']
        item_df = pd.read_csv(
                    os.path.join(dataset_dir, 'movies.csv'),
                    header=0,
                    sep=','
                )
        # genres
        item_df['genres'] = item_df['genres'].map(lambda x: x.split('|'))
        item_df['genres'] = item_df['genres'].map(lambda x: x if len(x) > 0 else ['[PAD]'])
        item_df['genres'] = item_df['genres'].map(lambda x: '|||'.join(x))

        # summary
        item_summary = json.load(open(os.path.join(dataset_dir, 'sampled_summaries.json')))
        item_df['summary'] = item_df['movieId'].map(lambda x: item_summary.get(str(x), ''))

        # write item information to csv
        item_df[item_fields].to_csv(
                                os.path.join(dataset_dir, 'meta_item.csv'),
                                sep='\t',
                                header=True,
                                index=False,
                                columns=item_fields
                            )
        
        # write ineractions to csv
        inter_fields = ['userId', 'movieId', 'rating', 'timestamp']
        inter_df = pd.read_csv(
                        os.path.join(dataset_dir, 'sampled_ratings.csv'), 
                        header=0,
                        sep=','
                    )
        inter_df.to_csv(
            os.path.join(dataset_dir, 'processed_ratings.csv'),
            header=True,
            index=False,
            sep='\t',
        )

    elif dataset == 'steam':
        from polyglot.detect import Detector

        item_fields = ['product_id', 'short_description', 'price', 'release_date', 'supported_languages', 'genres', 'categories', 'mac', 'windows']
        with open(os.path.join(dataset_dir, 'games.json'), 'r', encoding='utf-8') as f:
            item_data = json.loads(f.read())
        item_df = defaultdict(list)
        for product_id, item in item_data.items():
            datum = {'product_id': product_id}
            try:
                # filter non english short_description
                lang = sorted(Detector(item['short_description']).languages, key=lambda x: x.confidence, reverse=True)[0].name
                if lang == 'English':
                    datum['short_description'] = item['short_description']
                else:
                    continue
                
                # group price to buckets
                price = item['price']
                if price <= 20.0:
                    datum['price'] = str(int(price))
                elif price <= 30:
                    datum['price'] = '(20, 30]'
                elif price <= 40:
                    datum['price'] = '(30, 40]'
                elif price <= 60:
                    datum['price'] = '(40, 60]'
                elif price <= 100:
                    datum['price'] = '(60, 100]'
                elif price <= 300:
                    datum['price'] = '(100, 300]'
                else:
                    datum['price'] = '> 300'

                # group release date to buckets by year; keep years in 2006~2023
                year = item['release_date'].split(' ')[-1]
                if year < '2006' or year > '2023':
                    continue
                datum['release_date'] = year

                # clean supported langugage; pad if no category; join categories with sep
                datum['supported_languages'] = []
                for sl in item['supported_languages']:
                    sl = sl.replace(
                            '[b][/b]', ''
                        ).replace(
                            '&amp;lt;strong&amp;gt;&amp;lt;/strong&amp;gt;', ''
                        ).replace(
                            '&amp;lt;br /&amp;gt;&amp;lt;br /&amp;gt;', ''
                        ).replace(
                            '(text only)', ''
                        ).replace(
                            '(full audio)', ''
                        ).replace(
                            '(all with full audio support)', ''
                        ).replace(
                            'English Dutch', ''
                        ).replace(
                            ',#lang_français', ''
                        ).replace(
                            ';', ''
                        ).strip()
                    
                    if ',' in sl:
                        sl = sl.split(',')
                    elif '\r\n' in sl:
                        sl = sl.split('\r\n')
                    else:
                        sl = [sl]
                    datum['supported_languages'] += sl
                if len(datum['supported_languages']) == 0:
                    datum['supported_languages'] = ['[PAD]']
                datum['supported_languages'] = '|||'.join(datum['supported_languages'])

                # pad if no genre; join genres with sep
                datum['genres'] = item['genres']
                if len(datum['genres']) == 0:
                    datum['genres'] = ['[PAD]']
                datum['genres'] = '|||'.join(datum['genres'])

                # pad if no category; join categories with sep
                datum['categories'] = item['categories']
                if len(datum['categories']) == 0:
                    datum['categories'] = ['[PAD]']
                datum['categories'] = '|||'.join(datum['categories'])

                # ensure mac is bool
                if item['mac'] not in [True, False]:
                    continue
                datum['mac'] = str(item['mac'])

                # ensure windows is bool
                if item['windows'] not in [True, False]:
                    continue
                datum['windows'] = str(item['windows'])
                
                for f in item_fields:
                    item_df[f].append(datum[f])

            except Exception as e:
                print(e)
                
        item_df = pd.DataFrame(item_df)

        # write item information to csv
        item_df.to_csv(
                os.path.join(dataset_dir, 'meta_item.csv'),
                sep='\t',
                header=True,
                index=False,
                columns=item_fields
            )

        # write interactions to csv
        inter_fields = ['username', 'product_id', 'date']
        inter_df = get_data_from_json_by_line(
                    json_file_path=os.path.join(dataset_dir, 'steam_reviews.json'),
                    fields=inter_fields
                )
        inter_df.to_csv(
                    os.path.join(dataset_dir, 'processed_steam_reviews.csv'),
                    sep='\t',
                    header=True,
                    index=False,
                    columns=inter_fields
                )

    else:
        raise ValueError(f'Expect `amazon-*`, `movielens`, `steam`, but got {dataset}.')
            


def process_by_recstudio(dataset, data_config_path):
    data_conf = parser_yaml(data_config_path)
    dataset = SeqDataset(name=dataset, config=data_conf)
    dataset.inter_feat.sort_values(by=[dataset.ftime], inplace=True)
    if data_conf['binarized_rating_thres'] is not None:
        dataset._binarize_rating(data_conf['binarized_rating_thres'])
    return dataset


def negative_sample_and_split(dataset, val=False, max_behavior_len=1e5):

    def sample_a_negative(pos_list, num_iid):
        while True:
            neg_id = random.randint(0, num_iid - 1)
            if neg_id not in pos_list:
                return neg_id
    
    trn_set = []
    val_set = []
    tst_set = []
    num_iid = dataset.num_values(dataset.fiid)
    for uid, hist in tqdm(dataset.inter_feat.groupby(dataset.fuid)):
        pos_list = hist[dataset.fiid].tolist()
        for i in range(1, len(pos_list)):
            if i > max_behavior_len:
                start = i - max_behavior_len
            else:
                start = 0

            u_bh = pos_list[start : i]
            pos_iid = pos_list[i]
            neg_iid = sample_a_negative(pos_list, num_iid)
            if val:
                if i < len(pos_list) - 2:
                    trn_set.append((uid, u_bh, pos_iid, 1.0))
                    trn_set.append((uid, u_bh, neg_iid, 0.0))
                elif i == len(pos_list) - 2:
                    val_set.append((uid, u_bh, pos_iid, 1.0))
                    val_set.append((uid, u_bh, neg_iid, 0.0))
                else:
                    tst_set.append((uid, u_bh, pos_iid, 1.0))
                    tst_set.append((uid, u_bh, neg_iid, 0.0))
            else:
                if i < len(pos_list) - 1:
                    trn_set.append((uid, u_bh, pos_iid, 1.0))
                    trn_set.append((uid, u_bh, neg_iid, 0.0))
                else:
                    tst_set.append((uid, u_bh, pos_iid, 1.0))
                    tst_set.append((uid, u_bh, neg_iid, 0.0))

    random.shuffle(trn_set)
    return trn_set, val_set, tst_set


def no_sample_and_split(dataset, max_behavior_len=1e5, split_ratio=[0.8, 0.0, 0.2]):
    trn_set = []
    val_set = []
    tst_set = []
    for uid, hist in tqdm(dataset.inter_feat.groupby(dataset.fuid)):
        u_trn_set = []
        u_val_set = []
        u_tst_set = []

        all_list = hist[dataset.fiid].tolist()
        all_y = hist[dataset.frating].tolist()
        num_pos_bh = [0] + np.cumsum(all_y[:-1]).tolist()

        num_nonzero = sum([1 if _ > 0 else 0 for _ in num_pos_bh])
        splits = np.outer(
                    num_nonzero, 
                    split_ratio
                ).astype(np.int32).flatten().tolist()
        splits[0] = num_nonzero - sum(splits[1:])
        for i in range(1, len(split_ratio)):
            if (split_ratio[-i] != 0) & (splits[-i] == 0) & (splits[0] > 1):
                splits[-i] += 1
                splits[0] -= 1


        for i, n_p in enumerate(num_pos_bh):
            if n_p == 0:
                continue

            if n_p > max_behavior_len:
                start = num_pos_bh.index(n_p - max_behavior_len)
            else:
                start = 0

            u_bh = []
            for iid, y in zip(all_list[start : i], all_y[start : i]):
                if y == 1.0:
                    u_bh.append(iid)
            iid = all_list[i]
            y = all_y[i]

            if len(u_trn_set) < splits[0]:
                u_trn_set.append((uid, u_bh, iid, y))
            elif len(u_val_set) < splits[1]:
                u_val_set.append((uid, u_bh, iid, y))
            else:
                u_tst_set.append((uid, u_bh, iid, y))

        trn_set += u_trn_set
        val_set += u_val_set
        tst_set += u_tst_set

    random.shuffle(trn_set)
    return trn_set, val_set, tst_set



if __name__ == '__main__':
    seed_everything(42)
    dataset = sys.argv[1]

    if dataset == 'amazon-toys':
        dataset_dir = os.path.join(work_dir, 'data/Toys_and_Games')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('amazon', os.path.join(dataset_dir, 'Toys_and_Games.yaml'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = negative_sample_and_split(dataset, max_behavior_len=max_behavior_len)

    elif dataset == 'amazon-sports':
        dataset_dir = os.path.join(work_dir, 'data/Sports_and_Outdoors')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('amazon', os.path.join(dataset_dir, 'Sports_and_Outdoors'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = negative_sample_and_split(dataset, max_behavior_len=max_behavior_len)

    elif dataset == 'amazon-books':
        dataset_dir = os.path.join(work_dir, 'data/Books')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('amazon', os.path.join(dataset_dir, 'Books.yaml'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = negative_sample_and_split(dataset, max_behavior_len=max_behavior_len)

    elif dataset == 'amazon-clothing':
        dataset_dir = os.path.join(work_dir, 'data/Clothing_Shoes_and_Jewelry')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('amazon', os.path.join(dataset_dir, 'Clothing_Shoes_and_Jewelry.yaml'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = negative_sample_and_split(dataset, max_behavior_len=max_behavior_len)
    

    elif dataset == 'steam':
        dataset_dir = os.path.join(work_dir, 'data/Steam')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('steam', os.path.join(dataset_dir, 'Steam.yaml'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = negative_sample_and_split(dataset, max_behavior_len=max_behavior_len)

    elif dataset == 'movielens':
        dataset_dir = os.path.join(work_dir, 'data/MovieLens')
        adapt2recstudio(dataset, dataset_dir)
        extract_item_description(dataset, dataset_dir)
        dataset = process_by_recstudio('movielens', os.path.join(dataset_dir, 'MovieLens.yaml'))
        max_behavior_len = 20
        trn_set, val_set, tst_set = no_sample_and_split(dataset, max_behavior_len=max_behavior_len, split_ratio=[0.8, 0.0, 0.2])

    else:
        raise ValueError(f'Expect dataset to be `amazon-*`, `steam` or `movielens`, but got {dataset}.')
    

    with open(os.path.join(dataset_dir, f'truncated{max_behavior_len}_dataset.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
