import socket               # 导入 socket 模块
import numpy as np


from kaldi.matrix import Vector, Matrix, DoubleMatrix, DoubleSubMatrix


from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.pitch import PitchExtractionOptions, ProcessPitchOptions, compute_and_process_kaldi_pitch
from kaldi.transform.cmvn import Cmvn

from kaldi.feat.online import OnlineFbank, OnlineCmvnOptions, OnlineCmvn, OnlineCmvnState, OnlineAppendFeature
from kaldi.feat.pitch import OnlinePitchFeature, PitchExtractionOptions, OnlineProcessPitch, ProcessPitchOptions

from cmvn import CMVN, UtteranceCMVN

import torch
from torch import Tensor
from model.online_encoder import encoder_for
import json


def load_model():
    with open('./model/model.json', 'r') as f:
        dic = json.load(f)
        dic = {i: char for i, char in enumerate(dic[2]['char_list'])}

    state_dict = torch.load(
        './model/lstm_down_2x2att', map_location=torch.device('cpu'))
    state_dict = {k[4:]: v for k, v in state_dict.items() if k.split(
        '.')[0] == 'enc' or k.split('.')[0] == 'ctc'}

    model = encoder_for()
    model.load_state_dict(state_dict)
    model.eval()

    print('model loaded')
    return model, dic


def ctc_merge(src):
    new = []
    prev = None
    for char in src:
        if char != prev and char != '<blank>':
            new.append(char)
        prev = char

    return new



#######kaldi feature###########

fbank_bins = 80
sr = 16000
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
feats_cmvn = OnlineCmvn(CMVN_opts, cmvn_stats, feats)

# cmvn = CMVN(
#     './cmvn.ark', norm_vars=True)

model, dic = load_model()
model_chunk = 120

last_num = 0


s = socket.socket()         
host = socket.gethostname()  
port = 12345                
s.bind(('0.0.0.0', port))        
print(host)

s.listen(5)                 

import sys

mark=True
while True:
    states = None
    output = []
    c, addr = s.accept()    
    print('连接地址：', addr)
    while True:
        data = c.recv(2048)
        if not data:
            break
        audiodata = np.frombuffer(data, dtype=np.int16)


        fbank.accept_waveform(sr, Vector(audiodata))
        pitch_src.accept_waveform(sr, Vector(audiodata))

        num_frames_ready = feats_cmvn.num_frames_ready()

        if num_frames_ready - last_num > model_chunk:
            feats_cmvn.freeze(0)
            feature_data = Matrix(model_chunk, 83)
            
            feats_cmvn.get_frames(
                [last_num+i+1 for i in range(model_chunk)], feature_data)

            last_num = last_num + model_chunk

            feature_data = feature_data.numpy()
            # feature_data = cmvn(feature_data)

            feature_data = feature_data.reshape((1, -1, 83))

            result, states = model(Tensor(feature_data), [
                                feature_data.shape[1]], prev_states=states)

            result = torch.argmax(result, -1)[0].numpy().tolist()
            output += result

            result = ctc_merge([dic[r] for r in output])
            # print(result)
            if mark:
                print(''.join(result)+"+", end="\r")
            else:
                print(''.join(result)+"-", end="\r")

            mark = not mark
            sys.stdout.flush()
            

    c.close()                
