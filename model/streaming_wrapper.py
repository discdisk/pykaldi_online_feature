from torch import Tensor
from torch import cat as concat
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
class stream(object):
    def __init__(self):
        self.states={}
        self.funcs={}
        self.outs={}

    def add(self, name, func):
        self.states[name] = None
        self.funcs[name] = func
        self.outs[name]= Tensor([])


    def __call__(self, in_data, name, sub=0):
        func=self.funcs[name]
        print(in_data.shape)
        ys, hx = func(in_data,self.states[name])

        ys, ilens = pad_packed_sequence(ys, batch_first=True)


        self.states[name] = hx

        ys = concat((self.outs[name], ys))
        length = ys.shape[1] 
        if length>=sub:
            self.outs[name] = ys[:,length-length//2*2:]
            return ys[:,:length-length//2*2], ilens

        return None, None

    