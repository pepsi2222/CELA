from __future__ import print_function
from __future__ import division

import os
import sys
from io import open
import time
import shutil
import math
import argparse
from datetime import date
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import psutil
import pickle
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from typing import Union, Optional
import json
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_util.util import load_config
from generate_app_features import generate_app_features_file

print(__doc__)
parser = argparse.ArgumentParser()
parser.add_argument("day_start", help="process data from this date")
parser.add_argument("day_end", help="process data up to this date")
parser.add_argument("imei_feature", nargs='?', default="false", help="whether to add imei")
parser.add_argument("--day_end_as_testset",
                    help="use data in day_end as test set",
                    action="store_true")
parser.add_argument("--single_process",
                    help="process data in serial(for debug usage)",
                    action="store_true")
parser.add_argument("--incremental", help="incremental learning")
parser.add_argument("--config_task_file", help="config file of model")
parser.add_argument("--data_dir", type=str, help="data dir ")
parser.add_argument("--target_dir", type=str, help="target dir ")

args = parser.parse_args()
print('Incremental model', args.incremental)
mtp_enable = True if args.data_dir else False
config = load_config(args.config_task_file, mtp_enable=mtp_enable)
if config.MTP:
    config.DATA_DIR = args.data_dir
    config.TARGET_DIR = args.target_dir


def id_to_tfrecord(part_num, idx3, chunk, line_sz, key, out_dir, max_len):
    chunk = chunk.values
    chunk = chunk.astype('int32')
    num_lines = chunk.shape[0]
    ids = chunk[:, config.LABEL:max_len + config.LABEL]
    weights = chunk[:, max_len + config.LABEL:]
    if config.LABEL == 1:
        labels = chunk[:, 0]
    else:
        labels = chunk[:, 0:config.LABEL]
    num_of_line = num_lines // line_sz
    chunk_size = num_of_line * line_sz
    writer = tf.io.TFRecordWriter(os.path.join(out_dir, '%s_part_%d.tfrecord' % (key, part_num + idx3)))
    for line_index in range(num_of_line):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=labels[line_index * line_sz:(line_index + 1) * line_sz, ])),
            "feat_ids":
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=ids[line_index * line_sz:(line_index + 1) * line_sz, ].reshape(-1))),
            "feat_vals":
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=weights[line_index * line_sz:(line_index + 1) * line_sz, ].reshape(-1)))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    del ids, weights, labels
    return chunk_size


def mp_id_to_tfrecord(part_num, in_file, key, block_sz, line_sz, out_dir, max_len, num_core=10):
    print('Transferring id {} data into tfrecord format...'.format(key))
    reader = pd.read_csv(in_file, chunksize=block_sz,
                         header=None, dtype=np.int32, sep=' ')
    line_num = 0
    num_lines = []
    pool = mp.Pool(processes=num_core)
    for idx1, chunk in enumerate(reader):
        num_line = pool.apply_async(id_to_tfrecord, (part_num, idx1, chunk, line_sz, key, out_dir, max_len))
        num_lines.append(num_line)
    pool.close()
    pool.join()
    for num_line in num_lines:
        line_num += num_line.get()
        print(line_num)
    return idx1 + 1, line_num


def str_to_bool(flag):
    return True if flag.lower() == 'true' else False


def get_memory_usage():
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    return memory_usage


def id_to_hdf(id_to_hdf_argv, dense_argv):
    part_num, idx2, chunk, key, out_dir = id_to_hdf_argv
    dense_num, dense_min_list, dense_max_list = dense_argv
    label_loc = config.LABEL
    group_index = config.group_id_index

    data_x = chunk.loc[:, chunk.columns[config.data_x_start_index:]]

    if label_loc == 1:
        data_y = chunk.loc[:, chunk.columns[0]]
    else:
        data_y = chunk.loc[:, chunk.columns[0:config.LABEL]]
        
    for idx in range(dense_num):
        data_x.loc[:, data_x.columns[[idx]]] = data_x.loc[:, data_x.columns[[idx]]].fillna(0)
        if dense_max_list[idx] is not None:
            invalid_idx = data_x.index[data_x.loc[:, data_x.columns[idx]] > dense_max_list[idx]]
            data_x.loc[invalid_idx, data_x.columns[[idx]]] = 0.0
        if dense_min_list[idx] is not None:
            invalid_idx = data_x.index[data_x.loc[:, data_x.columns[idx]] < dense_min_list[idx]]
            data_x.loc[invalid_idx, data_x.columns[[idx]]] = 0.0
    
    with h5py.File(os.path.join(out_dir,
                                key + '_input_part_' + str(part_num + idx2) + '.h5'), 'w') as feature_file:
        feature_file.create_dataset('feat', data=data_x, dtype=np.float32)

    with h5py.File(os.path.join(out_dir,
                                key + '_output_part_' + str(part_num + idx2) + '.h5'), 'w') as feature_file:
        feature_file.create_dataset('label', data=data_y)
    print("part {} finished".format(idx2))
    del data_x, data_y


def is_use_part_train_data(key, idx):
    return key == 'train' and idx >= config.LINES_NUM and config.LINES_NUM > 0


def error_callback(error):
    print(f"Error info: {error}")

    
def mp_id_to_hdf(mp_id_to_hdf_argv, dense_argv, num_core=10):
    part_num, in_file, key, block_sz, out_dir = mp_id_to_hdf_argv
    dense_num, dense_min_list, dense_max_list = dense_argv
    print('Transferring id {} data into hdf format...'.format(key))
    reader = pd.read_csv(in_file, chunksize=block_sz, header=None, sep=' ')
    pool = mp.Pool(processes=num_core)
    for idx, chunk in enumerate(reader):
        if is_use_part_train_data(key, idx):
            print("part {} break".format(idx))
            break
        pool.apply_async(id_to_hdf,
                         ((part_num, idx, chunk, key, out_dir), (dense_num, dense_min_list, dense_max_list)), 
                         error_callback=error_callback)
    pool.close()
    pool.join()
    if is_use_part_train_data(key, idx):
        return idx
    else:
        return idx + 1


def check_id_files(rawid_files):
    if not rawid_files or len(rawid_files) == 0:
        raise ValueError("train files is empty! Please check whether the date is correct!")
    for rawid in rawid_files:
        rawid_size = int(os.path.getsize(rawid) / 1024)
        if config.RAWID_SIZE_LOW != 0 and rawid_size < config.RAWID_SIZE_LOW:
            raise ValueError("the size of file '{0}' is small :{1} KB".format(rawid, rawid_size))
        if config.RAWID_SIZE_HIGH != 0 and rawid_size > config.RAWID_SIZE_HIGH:
            raise ValueError("the size of file '{0}' is big :{1} KB".format(rawid, rawid_size))


def remove_line(delete_lines, del_file_path):
    if len(delete_lines) > 0:
        with open(del_file_path, 'r') as fr:
            all_lines = fr.readlines()

        with open(del_file_path + '_bak', 'w') as fw:
            for line_idx, cur_line in enumerate(all_lines):
                if line_idx not in delete_lines:
                    fw.write(cur_line)
            fw.truncate()

        os.remove(del_file_path)
        os.rename(del_file_path + '_bak', del_file_path)


def dense_feat_statistic(dense_num, in_files, key, block_sz):
    dense_process, dense_min_list, dense_max_list, n_bins = \
        config.DENSE_PROCESS, config.DENSE_MIN_LIST, config.DENSE_MAX_LIST, config.DENSE_N_BIN_LIST
        
    in_dense_processors = []
    for idx, x in enumerate(dense_process):
        if x == 'min_max':
            in_dense_processors.append(MinMaxScaler())
        elif x == 'standard':
            in_dense_processors.append(StandardScaler())
        elif x == 'uniform':
            in_dense_processors.append(
                KBinsDiscretizer(n_bins=n_bins[idx], encode='ordinal', strategy='uniform')
                )
        elif x == 'quantile':
            in_dense_processors.append(
                KBinsDiscretizer(n_bins=n_bins[idx], encode='ordinal', strategy='quantile')
                )
        elif x == 'kmeans':
            in_dense_processors.append(
                KBinsDiscretizer(n_bins=n_bins[idx], encode='ordinal', strategy='kmeans')
                )
        elif x is None:
            in_dense_processors.append(None)
        else:
            raise ValueError(f'{x} is not a valid dense feature process method.')
    
    label_loc = config.LABEL
    dense_col = [None for _ in range(dense_num)]
    for in_file in in_files:
        print(f'start to count {in_file}')
        
        reader = pd.read_csv(in_file, chunksize=block_sz, header=None, dtype=np.float32, sep=' ')
        for idx, chunk in enumerate(reader):
            if is_use_part_train_data(key, idx):
                print("part {} break".format(idx))
                break
            data_x = chunk.loc[:, chunk.columns[label_loc:]]
            if label_loc == 1:
                data_y = chunk.loc[:, chunk.columns[0]]
            else:
                if config.GROUP_AUC:
                    data_y = chunk.loc[:, chunk.columns[label_loc - 1]]
                else:
                    data_y = chunk.loc[:, chunk.columns[0:config.LABEL]]
            
            for j in range(dense_num):
                data_x.loc[:, data_x.columns[[j]]] = data_x.loc[:, data_x.columns[[j]]].fillna(0)
                if dense_max_list[j] is not None:
                    invalid_idx = data_x.index[data_x.loc[:, data_x.columns[j]] > dense_max_list[j]]
                    data_x.loc[invalid_idx, data_x.columns[[j]]] = 0.0
                if dense_min_list[j] is not None:
                    invalid_idx = data_x.index[data_x.loc[:, data_x.columns[j]] < dense_min_list[j]]
                    data_x.loc[invalid_idx, data_x.columns[[j]]] = 0.0
                
                if dense_process[j] in ['min_max', 'standard']:
                    dense_values = data_x.iloc[:, [j]].fillna(0)
                    dense_values = dense_values.to_numpy()
                    in_dense_processors[j].partial_fit(dense_values)
                elif dense_process[j] in ['uniform', 'quantile', 'kmeans']:
                    dense_values = data_x.iloc[:, [j]].fillna(0)
                    dense_values = dense_values.to_numpy()
                    sample_idx = np.random.choice(
                        len(dense_values), size=int(0.1 * len(dense_values)), replace=False)
                    dense_values = dense_values[sample_idx]
                    if dense_col[j] is None:
                        dense_col[j] = dense_values
                    else:
                        dense_col[j] = np.concatenate([dense_col[j], dense_values], 
                                                    axis=0)
    
    for idx in range(dense_num):
        if dense_process[idx] in ['uniform', 'quantile', 'kmeans']:
            in_dense_processors[idx].fit(dense_col[idx])
                    
    return in_dense_processors


def dense_feature_process(dense_num):
    
    if config.DENSE_PROCESS is None:
        config.DENSE_PROCESS = [None for _ in range(dense_num)]
    if not (isinstance(config.DENSE_PROCESS, list) and len(config.DENSE_PROCESS) == dense_num):
        raise ValueError(f'DENSE_PROCESS must be list and the length of it must be dense length.')
    
    if dense_num != 0:
        dense_dir = os.path.join(config.TARGET_DIR, 'dense_statistics')
        if not os.path.exists(dense_dir):
            os.makedirs(dense_dir)
        print('start dense features statistic')
        if len(config.DENSE_PROCESS) != dense_num:
            raise ValueError('The length of DENSE_PROCESS must equal to dense_length')
        dense_processors = dense_feat_statistic(dense_num, in_files=train_files, key='train', block_sz=block_size)
        
        dense_offsets = [0 for _ in dense_processors]
        bins_cnt = 0
        for idx, dense_processor in enumerate(dense_processors):
            if config.DENSE_PROCESS[idx] == 'min_max':
                print(f'min : {dense_processor.data_min_}, max : {dense_processor.data_max_}')
            elif config.DENSE_PROCESS[idx] == 'standard':
                print(f'min : {dense_processor.mean_}, max : {dense_processor.var_}')
            elif config.DENSE_PROCESS[idx] in ['uniform', 'quantile', 'kmeans']:
                print(f'strategy : {config.DENSE_PROCESS[idx]}, number of bins : {dense_processor.n_bins_[0]}')
                print(f'bin edges : {dense_processor.bin_edges_}')
                dense_offsets[idx] = bins_cnt
                bins_cnt += dense_processor.n_bins_[0]
            elif config.DENSE_PROCESS[idx] is None:
                print('None')
        print('dense features statistic ends')
        
        print('start to save dense features statistics')
        statistic_list = []
        for idx, dense_processor in enumerate(dense_processors):
            if config.DENSE_PROCESS[idx] == 'min_max':
                statistic_list.append(
                    {
                        'method': 'min_max',
                        'min': float(dense_processor.data_min_[0]), 
                        'max': float(dense_processor.data_max_[0])
                    })
            elif config.DENSE_PROCESS[idx] == 'standard':
                statistic_list.append(
                    {
                        'method': 'standard',
                        'mean': float(dense_processor.mean_[0]),
                        'scale': float(dense_processor.scale_[0])
                    })
            elif config.DENSE_PROCESS[idx] in ['uniform', 'quantile', 'kmeans']:
                statistic_list.append(
                    {
                        'method': 'bins', 
                        'edges': list(dense_processor.bin_edges_[0])
                    })
            elif config.DENSE_PROCESS[idx] is None:
                statistic_list.append({'method': 'none'})
        print(f'statistic list : {statistic_list}')
        statistic_json_str = json.dumps(statistic_list, indent=4)
        with open(os.path.join(dense_dir, f'{args.day_end}-dense_statistic.json'), 'w') as fff:
            fff.write(statistic_json_str)
        print('dense features statistics are saved')
        
        
if __name__ == '__main__':

    # step0 read data
    DATA_TYPE = config.DATA_TYPE
    block_size = config.LINES_PER_PART
    lines_num = config.LINES_NUM

    if config.MTP:
        train_data_path = os.path.join(config.DATA_DIR, 'train_data')
        if os.path.exists(train_data_path):
            dir_path = train_data_path
            raw_data_dir = os.path.join(dir_path, 'rawdata')
        else:
            dir_path = config.DATA_DIR
            raw_data_dir = config.DATA_DIR
        hdf_data_dir = os.path.join(config.TARGET_DIR, 'id_data')
        model_dir = os.path.join(config.TARGET_DIR, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    if not os.path.exists(raw_data_dir):
        raise ValueError("Data path not exists: {}".format(raw_data_dir))
    if not os.path.exists(hdf_data_dir):
        os.makedirs(hdf_data_dir)

    statistic_file = os.path.join(dir_path, 'statistic/statistic.info.' + args.day_end)
    fm_file = os.path.join(dir_path, 'feature_map/feature_map.' + args.day_end)
    print('dir: {}'.format(statistic_file))
    if config.MTP:
        if not os.path.exists(train_data_path):
            statistic_file = os.path.join(dir_path, args.day_end + '/merge_data.txt')
            fm_file = os.path.join(dir_path, args.day_end + '/merge_feature_map.txt')
        shutil.copyfile(statistic_file, config.TARGET_DIR + '/statistic.info')
        with open(statistic_file, 'r') as stat:
            stat_dict = dict()
            for line in stat:
                print(line.strip('\n').split(',')[1])
                stat_dict[line.strip('\n').split(',')[0]] = line.strip('\n').split(',')[1]
            try:
                config.max_length = int(stat_dict['max_length'])
            except KeyError:
                config.max_length = 0
        if config.HASHMAP is False:
            shutil.copyfile(fm_file, model_dir + '/featureMap.txt')
            shutil.copyfile(fm_file, os.path.join(model_dir, args.day_end + '.featureMap.txt'))
            # check featuremap copy
            with open(os.path.join(model_dir + '/featureMap.txt'), 'r') as fm:
                output_fm_lines = len(fm.readlines())
            with open(fm_file, 'r') as fm:
                fm_lines = len(fm.readlines())
            print('output fm lines', output_fm_lines, 'fm lines', fm_lines)
            if output_fm_lines != fm_lines:
                raise ValueError("output fm lines != fm lines")
    # check feature map line
    if config.HASHMAP is False and (config.FEATUREMAP_SIZE_LOW != 0 or config.FEATUREMAP_SIZE_HIGH != 0):
        with open(fm_file, 'r') as fm:
            fm_lines = len(fm.readlines())
        if config.FEATUREMAP_SIZE_LOW != 0 and fm_lines < config.FEATUREMAP_SIZE_LOW:
            raise ValueError("FeatureMap file has few lines: {}!".format(fm_lines))
        if config.FEATUREMAP_SIZE_HIGH != 0 and fm_lines > config.FEATUREMAP_SIZE_HIGH:
            raise ValueError("FeatureMap file has to much lines: {}!".format(fm_lines))


    with open(statistic_file, 'r') as stat:
        stat_dict = dict()
        for line in stat:
            print(line.strip('\n').split(',')[1])
            stat_dict[line.strip('\n').split(',')[0]] = line.strip('\n').split(',')[1]
        try:
            max_length = int(stat_dict['max_length'])
        except KeyError:
            max_length = 0
        try:
            dense_length = int(stat_dict['dense_length'])
        except KeyError:
            dense_length = 0  # common feature data have no thiwe
        try:
            feat_sizes = np.array(map(int, stat_dict['cat_sizes'].split(' ')))
        except KeyError:
            feat_sizes = 0
        try:
            num_features = np.sum(int(stat_dict['num_features'])) + 1
        except KeyError:
            num_features = 0
        try:
            multi_hot_flags = np.array(list(map(str_to_bool, stat_dict['multi_hot_flags'].split(' '))))
        except KeyError:
            multi_hot_flags = []
        try:
            multi_hot_len = int(stat_dict['multi_hot_len'])
        except KeyError:
            multi_hot_len = 0
    print("max length: %d\nnumber of features: %d\nfeat_sizes: %s" %
            (max_length, num_features, feat_sizes))
    print('multi_hot_flags:')
    print(multi_hot_flags)     

    y_start, m_start, d_start = map(int, [args.day_start[:4], args.day_start[4:6], args.day_start[6:]])
    y_end, m_end, d_end = map(int, [args.day_end[:4], args.day_end[4:6], args.day_end[6:]])
    date_start = date(y_start, m_start, d_start)
    date_end = date(y_end, m_end, d_end)
    dates = [date.fromordinal(ordinal) for ordinal in range(date_start.toordinal(), date_end.toordinal() + 1)]
    dates = [str(day).replace('-', '') for day in dates]
    print('dates specified: ', dates)

    files = []
    # variables to process imei
    imei_map, user_cnt = {}, 0
    for day in dates:
        all_file = os.path.join(raw_data_dir, config.RAW_FILE.replace('*', str(day)))
        if config.MTP and not os.path.exists(train_data_path):
            all_file = os.path.join(os.path.join(raw_data_dir, str(day)), "merge_rawId.txt")
        if os.path.isfile(all_file):
            print(all_file)
            files.append(all_file)
    
    if args.day_end_as_testset:
        train_files = files[:-1]
        test_files = files[-1:]
    else:
        train_files = files
        test_files = None
    test_files = files[-config.TEST_DAY_NUM:]
    if config.TRIAN_DAY_INTERVAL == 0:
        train_day_interval = len(files)
    else:
        train_day_interval = -config.TRIAN_DAY_INTERVAL
    train_files = files[:-config.TRIAN_DAY_INTERVAL]
    print('train files: ', train_files)
    print('test files: ', test_files)
    
    # variables to preprocess dense features
    # tfrecord is not supported temporarily
    
    if config.DENSE_MAX_LIST is None: 
        config.DENSE_MAX_LIST = [None for _ in range(dense_length)]
    if not (isinstance(config.DENSE_MAX_LIST, list) and len(config.DENSE_MAX_LIST) == dense_length):
        raise ValueError(f'DENSE_MAX_LIST must be list and the length of it must be dense length.')
    if config.DENSE_MIN_LIST is None: 
        config.DENSE_MIN_LIST = [None for _ in range(dense_length)]
    if not (isinstance(config.DENSE_MIN_LIST, list) and len(config.DENSE_MIN_LIST) == dense_length):
        raise ValueError(f'DENSE_MIN_LIST must be list and the length of it must be dense length.')
    
    if dense_length != 0 and config.DENSE_PROCESS is not None:
        dense_feature_process(dense_length)

    check_id_files(train_files)
    print('Got id data, initializing data set...')
    # empty the feature_data_dir and hdf_data_dir
    for f in os.listdir(hdf_data_dir):
        file_path = os.path.join(hdf_data_dir, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    print("the folders %s are now emptied." % hdf_data_dir)
    start_time = time.time()
    if config.CHECK_DATA:
        sample_number = 0
        feature_cover_ratio = []
        for train_file in train_files:
            with open(train_file, 'r') as file:
                for line in file:
                    sample_number += 1
                    line = line[:len(line) // 2 + 1]

    train_part_num = 0
    # step 1 transfer data to DATA_TYPE format
    
    for train_file in train_files:
        train_num_of_parts = mp_id_to_hdf((train_part_num, train_file, 'train', block_size, hdf_data_dir),
                                            (dense_length, config.DENSE_MIN_LIST, config.DENSE_MAX_LIST))
        train_part_num += train_num_of_parts
        print('Finished train file: ', train_file)
        if config.FILTER_LABEL_INDEX_FROM_MULTI is not None and config.HAS_TASK_MASK is not None:
            os.remove(train_file)
            print(f'{train_file} is removed')
    test_part_num = 0
    if test_files:
        for test_file in test_files:
            test_num_of_parts = mp_id_to_hdf((test_part_num, test_file, 'test', block_size, hdf_data_dir),
                                            (dense_length, config.DENSE_MIN_LIST, config.DENSE_MAX_LIST))
            test_part_num += test_num_of_parts
            print('Finished test file: ', test_file)
            if config.FILTER_LABEL_INDEX_FROM_MULTI is not None and config.HAS_TASK_MASK is not None:
                os.remove(test_file)
                print(f'{test_file} is removed')
    
    print("finish raw_to_hdf, time: %s sec" %
          (time.time() - start_time))
