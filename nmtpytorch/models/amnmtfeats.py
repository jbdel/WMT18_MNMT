# -*- coding: utf-8 -*-
import logging

import torch

from ..layers import TextEncoder, ConditionalMMDecoder, ConditionalMMDecoder_wmt18

from ..datasets import MultimodalDataset
from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class AttentiveMNMTFeatures(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'n_channels': 2048,         # depends on the features used
            'alpha_c': 0.0,             # doubly stoch. attention
            'trg_mul': False,
            'wmt18': False,
            'deep_gru': False,

        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

        self.decoder_func = ConditionalMMDecoder
        if self.opts.model['wmt18']:
            self.decoder_func = ConditionalMMDecoder_wmt18


    def setup(self, is_train=True):
        self.ctx_sizes['image'] = self.opts.model['n_channels']

        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            src_sorted_batches=True,
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # Create Decoder
        self.dec = self.decoder_func(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            trg_mul=self.opts.model['trg_mul'],
            options={
                'deep_gru': self.opts.model['deep_gru'],
            })

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch):
        feats = batch['image']
        if len(feats.size()) == 2:
            # if avg pool, size is of 2 : [batchsizexfeat_number]
            #need to do 1x32x2048
            feats = feats.unsqueeze(0)
        else:
            # Get features into (n,c,w*h) and then (w*h,n,c)
            feats = feats.view(
                (*feats.shape[:2], -1)).permute(2, 0, 1)

        return {
            'image': (feats, None),
            str(self.sl): self.enc(batch[self.sl]),
        }

    def forward(self, batch):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (1 - torch.cat(self.dec.alphas).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
