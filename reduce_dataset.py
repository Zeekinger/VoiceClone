import pickle 
import sys

if __name__ == '__main__':

    pkl_path = '../storage/data/LibriTTS/sr_24000_mel_norm/train.pkl'# /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/train.pkl
    output_path = '../storage/data/LibriTTS/sr_24000_mel_norm/train_128.pkl' # /groups/jjery2243542/data/LibriTTS/sr_24000_mel_norm/train_128.pkl
    segment_size = int(128) # 128

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    reduced_data = {key:val for key, val in data.items() if val.shape[0] > segment_size}

    with open(output_path, 'wb') as f:
        pickle.dump(reduced_data, f)
