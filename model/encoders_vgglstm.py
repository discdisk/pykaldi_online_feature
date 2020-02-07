import logging
import six

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

from torch.nn.utils.rnn import pad_sequence



class Local_Att(torch.nn.Module):
    """Local_Att module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str rnn_type: The RNN type
    """

    def __init__(self, hdim, odim, att_dim, att_heads, win_size):
        super(Local_Att, self).__init__()

        self.win_size  = win_size
        self.att_dim   = att_dim
        self.att_heads = att_heads
        self.hdim      = hdim
        self.odim      = odim

        self.query = torch.nn.Linear(self.hdim, att_dim*att_heads)
        self.key = torch.nn.Linear(hdim, att_dim*att_heads)
        self.value = torch.nn.Linear(hdim, hdim*att_heads)

        pad_size = win_size//2

        # (padding_left, padding_right, padding_top, padding_bottom)
        self.padding = torch.nn.ZeroPad2d((0, 0, pad_size, pad_size))
        self.last_out = torch.nn.Linear(self.att_heads*self.hdim, self.hdim)

        self.attend_ln = torch.nn.LayerNorm(self.hdim)
        # self.ctc_lo = torch.nn.Linear((conf['att_heads']+1)*conf['hdim'], conf['odim'])

        




    def forward(self, xs_pad, ilens):
        # print(ilens)
        if isinstance(xs_pad,tuple):
            xs_pad, residual_h1 = xs_pad
        else :
            residual_h1 = xs_pad

        batchSize ,seqLength, _ = xs_pad.shape
        # xs_pad shape: B x S x V
        # masks  shape: B x S x V
        masks = pad_sequence([torch.ones(i,1) for i in ilens],batch_first=True,padding_value=0).to(xs_pad.device)

        xs_pad = torch.unsqueeze(xs_pad,1)
        xs_pad = self.padding(xs_pad)
        xs_pad = torch.squeeze(xs_pad)

        if batchSize == 1:
           xs_pad = torch.unsqueeze(xs_pad, 0)

        if residual_h1 is not None:
            residual_h1 = torch.unsqueeze(residual_h1,1)
            residual_h1 = self.padding(residual_h1)
            residual_h1 = torch.squeeze(residual_h1)
            if batchSize == 1:
                residual_h1 = torch.unsqueeze(residual_h1, 0)

        masks = torch.unsqueeze(masks,1)
        masks = self.padding(masks)
        masks = -(masks-1).detach()
        masks = masks.byte()



        batchSize ,seqLength, _ = xs_pad.shape

        keys  = self.key(xs_pad)
        values = self.value(xs_pad)

        keys  = keys.view(batchSize,seqLength,self.att_heads,self.att_dim).transpose(1,2)
        values = values.view(batchSize,seqLength,self.att_heads,self.hdim).transpose(1,2)


        last_context = torch.zeros(batchSize, self.hdim).to(keys.device)
        result = []

        for i in range(seqLength-self.win_size+1):
            q = self.query(last_context)
            q = q.view(batchSize,self.att_heads, 1,self.att_dim)

            k =     keys[:,:,i:i+self.win_size,:]
            v =   values[:,:,i:i+self.win_size,:]
            mask = masks[:,:,i:i+self.win_size,:]

            context = self.cal_context(q,k,v,mask)

            context = F.dropout(self.last_out(context))


            # last_context = self.attend_ln(torch.cat((context,residual_h1[:,i+1+self.win_size//2,:]),dim=1))

            last_context = self.attend_ln(context+residual_h1[:,i+1+self.win_size//2,:])

            # last_context = self.ctc_lo(context)

            result.append(last_context)

        return torch.stack(result,1)

    def cal_context(self, q, k, v, mask):
        att_score = self.cal_score(q,k,mask)
        att_score = att_score.view(-1,1,att_score.shape[-2])


        att_weight = F.softmax(att_score, dim=-1)

        v = v.reshape(-1,v.shape[-2],v.shape[-1])


        context = torch.bmm(att_weight,v)
        # print(context.shape)
        context = context.view(mask.shape[0],-1)
        # print(context.shape)
        return context

    def cal_score(self, q, k, mask):
        attend = q+k

        score = torch.sum(torch.tanh(attend), dim=-1, keepdim=True)


        if mask is not None:
            score = score.masked_fill_(mask.bool(),-(np.inf))


        return score











class RNNP(torch.nn.Module):
    """RNN with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout, typ="blstm"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            rnn = torch.nn.LSTM(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir,
                                batch_first=True) if "lstm" in typ \
                else torch.nn.GRU(inputdim, cdim, dropout=dropout, num_layers=1, bidirectional=bidir, batch_first=True)
            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)
            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNNP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        elayer_states = []
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()
            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, states = rnn(xs_pack, hx=None if prev_state is None else prev_state[layer])
            elayer_states.append(states)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim


class RNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir) if "lstm" in typ \
            else torch.nn.GRU(idim, cdim, elayers, batch_first=True, dropout=dropout,
                              bidirectional=bidir)
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim , hdim)
        
        #Local_Att(hdim, odim, att_dim, att_heads, win_size)
        self.att = Local_Att(cdim, cdim, 200, 1, 13)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, [int(i*4/6) for i in ilens], batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)

        ys_pad = self.att(ys_pad,ilens)

        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)


        return xs_pad, ilens, states  # x: utt list of frame x dim


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, **kwargs):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                             xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 3, stride=3, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        return xs_pad, ilens, None  # no state in this layer


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ['lstm', 'gru', 'blstm', 'bgru']:
            logging.error("Error: need to specify an appropriate encoder architecture")

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNNP(get_vgg2l_odim(idim, in_channel=in_channel), elayers, eunits,
                                                     eprojs,
                                                     subsample, dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + 'P for encoder')
            else:
                self.enc = torch.nn.ModuleList([VGG2L(in_channel),
                                                RNN(int(get_vgg2l_odim(idim, in_channel=in_channel)*4/6), elayers, eunits,
                                                    eprojs,
                                                    dropout, typ=typ)])
                logging.info('Use CNN-VGG + ' + typ.upper() + ' for encoder')
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [RNNP(idim, elayers, eunits, eprojs, subsample, dropout, typ=typ)])
                logging.info(typ.upper() + ' with every-layer projection for encoder')
            else:
                self.enc = torch.nn.ModuleList([RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)])
                logging.info(typ.upper() + ' without projection for encoder')

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens, current_states


def encoder_for(args, idim, subsample):
    return Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs, subsample, args.dropout_rate)
