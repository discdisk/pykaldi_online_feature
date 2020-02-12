import pyaudio
import wave
from scipy.io import wavfile
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
        './model/lstm_down_3x2att', map_location=torch.device('cpu'))
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


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
sr = RATE = 16000
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

#######kaldi feature###########

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





##########pyaudio#####


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
result = []
last_num = 0


cmvn = CMVN(
    './cmvn.ark', norm_vars=True)

model, dic = load_model()
model_chunk = 180
print("* start recording")
from time import time
t=time()
states=None
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    

    
    data = stream.read(CHUNK)
    frames.append(data)
    audiodata = np.frombuffer(data, dtype=np.int16)


    fbank.accept_waveform(sr, Vector(audiodata))
    pitch_src.accept_waveform(sr, Vector(audiodata))

    num_frames_ready = feats.num_frames_ready()
    # print(num_frames_ready)

    
    # print(num_frames_ready)
    if num_frames_ready - last_num > model_chunk:
        feature_data = Matrix(model_chunk, 83)
        print(num_frames_ready-last_num)
    
        feats.get_frames([last_num+i+1 for i in range(model_chunk)], feature_data)

        last_num = last_num + model_chunk

        feature_data = cmvn(feature_data.numpy())

        feature_data = feature_data.reshape((1, -1, 83))

        result, states = model(Tensor(feature_data), [feature_data.shape[1]], prev_states=states)

        result = torch.argmax(result, -1)[0].numpy()

        result = ctc_merge([dic[r] for r in result])
        print(''.join(result))

    # print(numpydata.shape)
        

print("* done recording",time()-t)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# audiodata = np.frombuffer(b''.join(frames), dtype=np.int16)
# fbank.accept_waveform(sr, Vector(audiodata))
# pitch_src.accept_waveform(sr, Vector(audiodata))

# num_frames_ready = feats.num_frames_ready()
# feature_data = Matrix(num_frames_ready-last_num, 83)

# feats.get_frames([last_num+i+1 for i in range(num_frames_ready-last_num)], feature_data)


# feature_data = cmvn(feature_data.numpy())

# feature_data = feature_data.reshape((1, -1, 83))

# result, states = model(Tensor(feature_data), [feature_data.shape[1]], prev_states=states)

# result = torch.argmax(result, -1)[0].numpy()

# result = ctc_merge([dic[r] for r in result])
# print(''.join(result))
