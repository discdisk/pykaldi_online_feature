from kaldi.feat.fbank import Fbank, FbankOptions
from kaldi.feat.wave import WaveData
from scipy.io import wavfile
opts = FbankOptions()
sr=opts.frame_opts.samp_freq
print(sr)
fbank = Fbank(opts)
sr, sig = wavfile.read('old.wav')
print(sr)

from kaldi.matrix import Vector,Matrix
sig = Vector(sig)
# print(sig)
# y=fbank.compute_features(sig,sr,1)
# print(y)


from kaldi.feat.online import OnlineFbank

olfbank=OnlineFbank(opts)
olfbank.accept_waveform(sr, sig)
print(olfbank.frame_shift_in_seconds())
print(olfbank.num_frames_ready())
print(olfbank.input_finished())
x=Vector(13)
print(x)
print(olfbank.get_frame(899,x))
print(x)
print(olfbank.get_frame(900,x))
print(x)
print(olfbank.get_frame(901,x))
print(x)





