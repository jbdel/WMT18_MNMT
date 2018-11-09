# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..utils.nn import get_rnn_hidden_state
from . import Fusion, Attention, FF
from . import ConditionalDecoder

#simple pool trans + gate
class ConditionalMMDecoder_wmt18(ConditionalDecoder):
    """A conditional multimodal decoder with multimodal attention."""
    def __init__(self, fusion_type='concat',
                 trg_mul=False,
                 options=None,
                 **kwargs):
        super().__init__(**kwargs)


        self.trg_mul = trg_mul
        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))


        self.do_out1 = nn.Dropout(p=self.dropout_out)



        # Define (context) fusion operator

        self.pool_transform = FF(self.ctx_size_dict['image'], self.hidden_size, bias=True, activ="tanh")
        self.pool_transform_gate = FF(self.ctx_size_dict['image'], self.hidden_size, bias=True, activ="sigmoid")



        if options["deep_gru"]:
            self.dec3 = RNN(self.hidden_size + self.hidden_size + self.input_size, self.hidden_size)


        self.hid2out_gate  = FF(self.hidden_size, self.input_size,
                           bias_zero=True, activ='sigmoid')


        self.hid2out2      = FF(self.hidden_size, self.input_size,
                           bias_zero=True, activ='tanh')
        self.hid2out2_gate = FF(self.hidden_size, self.input_size,
                           bias_zero=True, activ='sigmoid')



        self.out2prob2 = FF(self.input_size, self.n_vocab)



        # Rename textual attention layer
        self.txt_att = self.att
        del self.att






    def f_next(self, ctx_dict, y, h):
        feats, _ = ctx_dict['image']
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        self.txt_alpha_t, z_t = self.txt_att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])


        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        gates = self.pool_transform_gate(feats).squeeze(0)
        c = self.pool_transform(feats).squeeze(0)
        c_t = torch.mul(c, gates)


        h3_c3 = self.dec3(torch.cat((c_t,z_t,y),dim=1), h2_c2)
        h3 = get_rnn_hidden_state(h3_c3)






        logit2 = self.hid2out(h2)
        logit2_gate = self.hid2out_gate(h2)
        log2 = torch.mul(logit2,logit2_gate)
        logit3 = self.hid2out2(h3)
        logit3_gate = self.hid2out2_gate(h3)
        log3 = torch.mul(logit3,logit3_gate)



        # Apply dropout if any
        if self.dropout_out > 0:
            log2 = self.do_out(log2)
            log3 = self.do_out1(log3)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim

        soft1 = self.out2prob(log2)
        soft2 = self.out2prob2(log3)

        log_p = F.log_softmax(torch.add(soft1,soft2), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h2_c2)
