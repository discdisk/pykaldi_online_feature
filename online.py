from scipy.io import wavfile
import numpy as np

from kaldi.matrix import Vector, Matrix, DoubleMatrix, DoubleSubMatrix


from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.pitch import PitchExtractionOptions, ProcessPitchOptions, compute_and_process_kaldi_pitch
from kaldi.transform.cmvn import Cmvn

from kaldi.feat.online import OnlineFbank, OnlineCmvnOptions, OnlineCmvn, OnlineCmvnState, OnlineAppendFeature
from kaldi.feat.pitch import OnlinePitchFeature, PitchExtractionOptions, OnlineProcessPitch, ProcessPitchOptions

from cmvn import CMVN, UtteranceCMVN

def ctc_merge(src):
    new = []
    prev = None
    for char in src:
        if char != prev and char != '<blank>':
            new.append(char)
        prev = char

    return new

def gen_feats(t):

    sr, sig = wavfile.read(
        './output.wav')

    print(sig.dtype)
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

    #cmvn
    CMVN_opts = OnlineCmvnOptions()
    CMVN_opts.normalize_variance = True


## feature exractor
    fbank = OnlineFbank(Fbank_opts)

    cmvn_stats = OnlineCmvnState()

    cmvn = Cmvn()
    cmvn.read_stats('./cmvn.ark')

    cmvn_stats = cmvn_stats.from_stats(cmvn.stats)

    



    pitch_src = OnlinePitchFeature(Pitch_opts)
    pitch = OnlineProcessPitch(Pitch_process_opts, pitch_src)

    feats = OnlineAppendFeature(fbank, pitch)

    # feats_cmvn = OnlineCmvn(CMVN_opts, cmvn_stats, feats)
    # feats_cmvn.freeze(0)

    result = []
    last_num = 0
    for i in range(int(t*sr/100)):
        x = Vector(83)
        fbank.accept_waveform(sr, sig[i*100:(i+1)*100])
        pitch_src.accept_waveform(sr, sig[i*100:(i+1)*100])

        num_frames_ready = feats.num_frames_ready()
        # print(num_frames_ready)
        if num_frames_ready > last_num:
            feats.get_frame(num_frames_ready, x)
            result.append(x.numpy())
            last_num=num_frames_ready

    return result



if __name__ == '__main__':
    import torch
    from torch import Tensor
    from model.online_encoder import encoder_for
    import json

    with open('./model/model.json', 'r') as f:
        dic = json.load(f)
        dic = {i: char for i, char in enumerate(dic[2]['char_list'])}

    state_dict = torch.load(
        './model/lstm_down_3x2att', map_location=torch.device('cpu'))
    state_dict = {k[4:]: v for k, v in state_dict.items() if k.split(
        '.')[0] == 'enc' or k.split('.')[0] == 'ctc'}

    model = encoder_for()
    model.load_state_dict(state_dict)
    model.eval()

    print('model loaded')

    cmvn = CMVN(
        './cmvn.ark', norm_vars=True)

    # cmvn = UtteranceCMVN(norm_means=True, norm_vars=True)

    feats = gen_feats(50)
    print(len(feats))
    feats=[np.array(feats[i*5:i*5+5]) for i in range(len(feats)//5)]
    feats = np.array([cmvn(feat) for feat in feats])

    feats = feats.reshape((1, -1, 83))


    print(feats.shape)


    states=None
    for i in range(feats.shape[1]//300): 
        result, states = model(Tensor(feats[:,i*300:i*300+300,:]), [feats[:,i*300:i*300+300,:].shape[1]], prev_states=states)

        result = torch.argmax(result, -1)[0].numpy()

        result = ctc_merge([dic[r] for r in result])
        print(''.join(result))
