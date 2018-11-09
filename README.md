# WMT18_MNMT
Solution from UMONS system at WMT 18

Clone the repository. This repository is a fork from the original [nmtpytorch](https://github.com/lium-lst/nmtpytorch/tree/master) framework.
Use 
```python
```

python setup.py develop

To install the framework in your python environement (virtualenv, for example).

To replicate results, simply train 5 models, and then evaluate :

```
output=model_out
nmtpy train -C examples/pool_multi30k-en-de-bpe10k.conf model.wmt18:True model.deep_gru:True train.save_path:./${output} && \
nmtpy train -C examples/pool_multi30k-en-de-bpe10k.conf model.wmt18:True model.deep_gru:True train.save_path:./${output} && \
nmtpy train -C examples/pool_multi30k-en-de-bpe10k.conf model.wmt18:True model.deep_gru:True train.save_path:./${output} && \
nmtpy train -C examples/pool_multi30k-en-de-bpe10k.conf model.wmt18:True model.deep_gru:True train.save_path:./${output} && \
nmtpy train -C examples/pool_multi30k-en-de-bpe10k.conf model.wmt18:True model.deep_gru:True train.save_path:./${output} && \
nmtpy translate $(ls ${output}/pool_multi30k-en-de-bpe10k/*.best.meteor.ckpt) -s val,test_2016_flickr,test_2017_flickr,test_2017_mscoco,test_2018_flickr -o ${output}
```

The deepgru model is located in nmtpytorch/layers/condmm_decoder_wmt18.py and is chosen thanks to the model.wmt18:True flag in 
nmtpytorch/models/amnmtfeats.py


