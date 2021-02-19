import pickle 
import librosa 
import sys
import glob 
import random
import os
from collections import defaultdict
import re
import numpy as np
import json
from tacotron.utils import get_spectrograms
import yaml

def read_speaker_info(speaker_info_path):
    speaker_ids = []
    with open(speaker_info_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            speaker_id = line.strip().split()[0]
            speaker_ids.append(speaker_id)
    return speaker_ids

def read_paths(root_dir, dset):
    paths = sorted(glob.glob(os.path.join(root_dir, f'{dset}/*/*/*.wav')))
    return paths

def get_speaker2path(root_dir, dset):
    speaker2path = defaultdict(lambda : [])
    for path in sorted(glob.glob(os.path.join(root_dir, f'{dset}/*/*/*.wav'))):
        filename = path.strip().split('/')[-1]
        speaker_id = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)\.wav', filename).groups()[0]
        speaker2path[speaker_id].append(path)
    return speaker2path

def spec_feature_extraction(wav_file):
    mel, mag = get_spectrograms(wav_file)
    return mel, mag

if __name__ == '__main__':

    data_dir = '../storage/data/raw/LibriTTS/'#sys.argv[1]  #raw_data_dir /groups/jjery2243542/data/raw/LibriTTS/
    output_dir = '../storage/data/LibriTTS/sr_24000_mel_norm/'#sys.argv[2] #data_dir /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/
    dev_proportion = float(0.05) #test_prop 0.05
    n_utts_attr = int(5000) #n_utts_attr 5000
    train_set ='train-clean-100'#sys.argv[5] #train_set train-clean-100
    #test_set = sys.argv[6] # dev-clean
    print(os.path.realpath(data_dir))
    paths = read_paths(data_dir, train_set)
    random.shuffle(paths)
    dev_data_size = int(len(paths) * dev_proportion)
    train_paths = paths[:-dev_data_size]
    dev_paths = paths[-dev_data_size:]
    #test_paths = read_paths(data_dir, test_set)
    print(f'{len(train_paths)} training data, {len(dev_paths)} dev data')

    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:
        for path in sorted(train_paths):
            filename = path.strip().split('/')[-1]
            f.write(f'{filename}\n')

    with open(os.path.join(output_dir, 'dev_files.txt'), 'w') as f:
        for path in sorted(dev_paths):
            filename = path.strip().split('/')[-1]
            f.write(f'{filename}\n')

    #with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:
    #    for path in sorted(test_paths):
    #        filename = path.strip().split('/')[-1]
    #        f.write(f'{filename}\n')

    for dset, paths in zip(['train', 'dev'], \
            [train_paths, dev_paths]):
        print(f'processing {dset} set, {len(paths)} files')
        data = {}
        output_path = os.path.join(output_dir, f'{dset}.pkl') # /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/train.pkl
        all_train_data = []
        for i, path in enumerate(paths):
            if i % 500 == 0 or i == len(paths) - 1:
                print(f'processing {i} files')
            filename = path.strip().split('/')[-1]
            mel, mag = spec_feature_extraction(path)
            data[filename] = mel
            if dset == 'train' and i < n_utts_attr:
                all_train_data.append(mel)
        if dset == 'train':
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {'mean': mean, 'std': std}
            with open(os.path.join(output_dir, 'attr.pkl'), 'wb') as f:
                pickle.dump(attr, f)
        for key, val in data.items():
            val = (val - mean) / std
            data[key] = val
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
