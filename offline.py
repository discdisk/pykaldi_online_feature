from scipy.io import wavfile
import numpy as np

from kaldi.matrix import Vector,Matrix, DoubleMatrix, DoubleSubMatrix

from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.pitch import PitchExtractionOptions, ProcessPitchOptions, compute_and_process_kaldi_pitch
from kaldi.transform.cmvn import Cmvn
def ctc_merge(src):
    new=[]
    prev=None
    for char in src:
        if char != prev and char != '<blank>':
            new.append(char)
        prev = char

    return new

def gen_feats(t):
    sr, sig = wavfile.read(
        '/mnt/c/Users/cjh06/OneDrive/Desktop/online_rec/test.wav')
    sig = Vector(sig[:t*sr])

    fbank_bins = 80
    # pipline

    # wave -> fbank+pitch -> cmvn -> feature



    ## make feature exraction options
    #fbank
    Fbank_opts = FbankOptions()
    Fbank_opts.frame_opts.samp_freq = sr
    Fbank_opts.mel_opts.num_bins = fbank_bins

    #pitch

    Pitch_opts = PitchExtractionOptions()
    Pitch_opts.samp_freq = sr
    Pitch_process_opts = ProcessPitchOptions()



    ## feature exractor
    fbank_exractor = Fbank(Fbank_opts)

    cmvn = Cmvn()
    cmvn.read_stats('/mnt/c/Users/cjh06/OneDrive/Desktop/online_rec/cmvn.ark')

    ##extraction

    fbank = fbank_exractor.compute_features(sig,sr,1)
    pitch = compute_and_process_kaldi_pitch(Pitch_opts, Pitch_process_opts, sig)


    feats = np.concatenate((fbank.numpy(),pitch.numpy()),axis = 1)
    feats = Matrix(feats)
    # cmvn.apply(feats, norm_vars=True)

    return feats.numpy()

if __name__=='__main__':
    import torch
    from torch import Tensor
    from model.encoders import encoder_for
    from cmvn import CMVN
    import json

    with open('/mnt/c/Users/cjh06/OneDrive/Desktop/online_rec/model/model.json', 'r') as f:
        dic = json.load(f)
        dic = {i:char for i,char in enumerate(dic[2]['char_list'])}



    state_dict = torch.load(
        '/mnt/c/Users/cjh06/OneDrive/Desktop/online_rec/model/lstm_down_3x2att', map_location=torch.device('cpu'))
    state_dict = {k[4:]: v for k, v in state_dict.items() if k.split('.')[0] == 'enc' or k.split('.')[0] == 'ctc'}

    model = encoder_for()
    model.load_state_dict(state_dict)
    model.eval()

    print('model loaded')

    feats=gen_feats(50).reshape(1, -1, 83)
    cmvn = CMVN(
        '/mnt/c/Users/cjh06/OneDrive/Desktop/online_rec/cmvn.ark', norm_vars=True)
    feats = cmvn(feats)

    print(feats.shape)

    result = model(Tensor(feats), [feats.shape[1]])
    result = torch.argmax(result,-1)[0].numpy()

    result = ctc_merge([dic[r] for r in result])
    print(''.join(result))



