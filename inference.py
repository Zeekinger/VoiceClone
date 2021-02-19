import torch.nn.functional as F
import yaml
import pickle
from model import AE,timefn
from utils import *
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms
import librosa 

class Inferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        #print(config)
        # args store other information
        self.args = args
        #print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)
    @timefn
    def load_model(self):
        print(f'Load model from {self.args.model}')
        self.model.load_state_dict(torch.load(f'{self.args.model}'))
        return
    @timefn
    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        #print(self.model)
        self.model.eval()
        return
    @timefn
    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out
    @timefn
    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec
    @timefn
    def denormalize(self, x):
        ## .pkl save the melspectrogram point mean and std
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    @timefn
    def inference_from_path(self):

        src_mel, _ = get_spectrograms(self.args.source)
        tar_mel, _ = get_spectrograms(self.args.target)
        #print(src_mel.shape,'   ',tar_mel.shape)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        #print(src_mel[:3,:3], '   ', tar_mel[:3,:3])
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)

        self.write_wav_to_file(conv_wav, self.args.output)
        return

if __name__ == '__main__':

    print("Is connect between machine")
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a',help='attr file path', default='attr/attr.pkl')
    parser.add_argument('-config', '-c',help='config file path', default='config.yaml')
    parser.add_argument('-model', '-m',help='model path', default='checkpoints/vctk_model.ckpt')
    parser.add_argument('-source', '-s',help='source wav path', default='test/source/normal.wav')
    parser.add_argument('-target', '-t',help='target wav path', default='test/target/slow.wav')
    parser.add_argument('-output', '-o',help='output wav path', default='converted_sound/normal2slow2.mp3')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=24000, type=int)
    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f)

    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()

    print("inference successful")
