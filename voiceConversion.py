from model import AE
from preprocess.tacotron.utils import get_spectrograms,melspectrogram2wav
import torch.nn.functional as F
import yaml
import pickle
from utils import *
from argparse import ArgumentParser
from scipy.io.wavfile import write

class VoiceConversion(object):
    def __init__(self,config,args):
        self.config = config
        self.args = args
        self.build_model()
        self.load_model()
        with open(self.args.attr,'rb') as f:
            self.attr = pickle.load(f)

    def build_model(self):
        self.model = cc(AE(self.config))
        self.model.eval()
        return

    def load_model(self):
        self.model.load_state_dict(torch.load(f'{self.args.model}'))
        return

    def normalize(self,x):
        m , s = self.attr['mean'], self.attr['std']
        res = (x - m) / s
        return res

    def denormalize(self,x):
        m , s = self.attr['mean'],self.attr['std']
        res = x * s + m
        return res

    def utt_make_frames(self,x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0)%frame_size
        if remains != 0:
            x = F.pad(x,(0,remains))
        out = x.view(1,x.size(0)//frame_size,frame_size * x.size(1)).transpose(1,2)
        return out

    def get_TargetSpeaker_Identity(self):
        tar_mel , _ = get_spectrograms(self.args.target)
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        tar_mel = self.utt_make_frames(tar_mel)
        emb = self.model.speaker_encoder(tar_mel)
        return emb

    def get_SourceSpeaker_Content(self):
        src_mel , _ = get_spectrograms(self.args.source)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        src_mel = self.utt_make_frames(src_mel)
        res , _ = self.model.content_encoder(src_mel)
        return res

    def get_Conversion_melSpectrogram(self,x,x_cond):
        conv = self.model.decoder(x,x_cond)
        conv = conv.transpose(1,2).squeeze(0)
        conv = conv.detach().cpu().numpy()
        conv = self.denormalize(conv)
        return conv

    def get_Concersion_Waveform(self,melSpectrogram):
        res = melspectrogram2wav(melSpectrogram)
        return res

    def save_Conversion_Waveform(self,wav_data,output_path):
        write(output_path,rate = self.args.sample_rate,data=wav_data)
        return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr','-a',default='attr/attr.pkl',help='attr file path')
    parser.add_argument('-config','-c',default='config.yaml',help='config file path')
    parser.add_argument('-model','-m',default='checkpoints/vctk_model.ckpt',help='model path')
    parser.add_argument('-source','-s',default='test/source/normal.wav',help='source wav path')
    parser.add_argument('-target','-t',default='test/target/slow.wav',help='target wav path')
    parser.add_argument('-output','-o',default='converted_sound/normal2slow.mp3',help='output wav path')
    parser.add_argument('-sample_rate','-sr',default=24000,type=int,help='sample rate')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    voiceconversion = VoiceConversion(config=config,args=args)

    emb = voiceconversion.get_TargetSpeaker_Identity()

    cont = voiceconversion.get_SourceSpeaker_Content()

    conv = voiceconversion.get_Conversion_melSpectrogram(cont,emb)

    conv = voiceconversion.get_Concersion_Waveform(conv)
     
    voiceconversion.save_Conversion_Waveform(conv,args.output)






