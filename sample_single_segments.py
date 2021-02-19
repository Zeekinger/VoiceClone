import json 
import pickle 
import sys
import os
import random

if __name__ == '__main__':

    pickle_path = '../storage/data/LibriTTS/sr_24000_mel_norm/dev.pkl' # /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/train.kpl
    sample_path = '../storage/data/LibriTTS/sr_24000_mel_norm/dev_sample_128.json'# /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/train_sample_128.json
    n_samples = int(10000) # 10000000
    segment_size = int(128) # 128

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # (utt_id, timestep, neg_utt_id, neg_timestep)
    samples = []

    # filter length > segment_size
    utt_list = [key for key in data]
    utt_list = sorted(list(filter(lambda u : len(data[u]) > segment_size, utt_list)))
    print(f'{len(utt_list)} utterances')
    sample_utt_index_list = random.choices(range(len(utt_list)), k=n_samples)


    for i, utt_ind in enumerate(sample_utt_index_list):
        if i % 500 == 0:
            print(f'sample {i} samples')
        utt_id = utt_list[utt_ind]
        t = random.randint(0, len(data[utt_id]) - segment_size)
        samples.append((utt_id, t))

    with open(sample_path, 'w') as f:
        json.dump(samples, f)

