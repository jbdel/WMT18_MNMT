![nmtpytorch](docs/logo.png?raw=true "nmtpytorch")

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the PyTorch fork of [nmtpy](https://github.com/lium-lst/nmtpy),
a sequence-to-sequence framework which was originally a fork of
[dl4mt-tutorial](https://github.com/nyu-dl/dl4mt-tutorial).

`nmtpytorch` is developed and tested on Python 3.6 and will not support
Python 2.x whatsoever.

## Citation

If you use **nmtpytorch**, you may want to cite the following [paper](https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf) although it was for the older Theano version:
```
@article{nmtpy2017,
  author    = {Ozan Caglayan and
               Mercedes Garc\'{i}a-Mart\'{i}nez and
               Adrien Bardet and
               Walid Aransa and
               Fethi Bougares and
               Lo\"{i}c Barrault},
  title     = {NMTPY: A Flexible Toolkit for Advanced Neural Machine Translation Systems},
  journal   = {Prague Bull. Math. Linguistics},
  volume    = {109},
  pages     = {15--28},
  year      = {2017},
  url       = {https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf},
  doi       = {10.1515/pralin-2017-0035},
  timestamp = {Tue, 12 Sep 2017 10:01:08 +0100}
}
```

## Release Notes

See [NEWS.md](NEWS.md).

## Installation

See [INSTALL.md](INSTALL.md).

## Usage Example

A [sample NMT configuration](examples/multi30k-en-de-bpe10k.conf) for
English-to-German Multi30k is provided which covers nearly all of the `[train]`
and `[model]` specific options to `NMT`.

After creating a configuration file for your own dataset that suits your need,
you can run the following command to start training:

```
nmtpy train -C <config file>
```

It is possible to override any configuration option through the command-line:

```
nmtpy train -C <config file> train.<opt>:<val> model.<opt>:<val> ...
```

### Differences compared to Theano-based nmtpy

The initial release aims to be (as much as) feature compatible with respect
to the latest `nmtpy` with some important changes as well.

#### New TensorBoard Support

If you would like to monitor training progress, you may want to install
[tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch). Note that
you will also need to install the actual TensorBoard server which is shipped
within Tensorflow in order to launch the visualization server.

Once the dependencies are installed, you need to define a log directory for
TensorBoard in the `configuration` file of your experiment to enable
TensorBoard logging. The logging frequency is the same as terminal logging
frequency, defined by `train.disp_freq` option (default: 30 batches).

```
[train]
..
tensorboard_dir: ~/tb_dir
```

![tensorboard](docs/tensorboard.png?raw=true "tensorboard")


#### A Single Command-Line Interface

Instead of shipping several tools for training, rescoring, translating, etc.
here we provide a single command-line interface `nmtpy` which implements
three subcommands `train`, `translate` and `test`.

**`nmtpy train`**

```
usage: nmtpy train [-h] -C CONFIG [-s SUFFIX] [-S] [overrides [overrides ...]]

positional arguments:
  overrides             (section).key:value overrides for config

optional arguments:
  -h, --help            show this help message and exit
  -C CONFIG, --config CONFIG
                        Experiment configuration file
  -s SUFFIX, --suffix SUFFIX
                        Optional experiment suffix.
  -S, --short           Use short experiment id in filenames.
```

**`nmtpy translate`**
```
usage: nmtpy translate [-h] [-n] [-b BATCH_SIZE] [-k BEAM_SIZE] [-m MAX_LEN]
                       [-a LP_ALPHA] [-d DEVICE_ID] (-s SPLITS | -S SOURCE) -o
                       OUTPUT
                       models [models ...]

positional arguments:
  models                Saved model/checkpoint file(s)

optional arguments:
  -h, --help            show this help message and exit
  -n, --disable-filters
                        Disable eval_filters given in config
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for beam-search
  -k BEAM_SIZE, --beam-size BEAM_SIZE
                        Beam size for beam-search
  -m MAX_LEN, --max-len MAX_LEN
                        Maximum seq. limit (Default: 200)
  -a LP_ALPHA, --lp-alpha LP_ALPHA
                        Apply length-penalty (Default: 0.)
  -d DEVICE_ID, --device-id DEVICE_ID
                        Select GPU device(s)
  -s SPLITS, --splits SPLITS
                        Comma separated splits from config file
  -S SOURCE, --source SOURCE
                        Comma-separated key:value pairs to provide new inputs.
  -o OUTPUT, --output OUTPUT
                        Output filename prefix
```

#### Experiment Configuration

The INI-style experiment configuration file format is slightly updated to
allow for future multi-task, multi-lingual setups in terms of data description.

Model-agnostic options are defined in `[train]` section while the options
that will be consumed by the model itself are defined in `[model]`.

An arbitrary number of parallel corpora with multiple languages can be defined
in `[data]` section. Note that you **need** to define at least
`train_set` and `val_set` datasets in this section for the training and
early-stopping to work correctly.

We recommend you to take a look at the provided sample
[configuration](examples/multi30k-en-de-bpe10k.conf) to have an idea about the file format.

#### Training a Model

We still provide a single, model-agnostic `mainloop` that handles everything
necessary to train, validate and early-stop a model.

#### Defining a Model

You just need to create a new file under `nmtpytorch/models` and define a
`class` by deriving it from `nn.Module`. The name of this new `class` will be the
`model_type` that needs to be written inside your configuration file. The next
steps are to:

 - Parse model options passed from the configuration file in `__init__()`
 - Define layers inside `setup()`: Each `nn.Module` object should be assigned
   as an attribute of the model (i.e. `self.encoder = ...`) in order for
   PyTorch to work correctly.
 - Create and store relevant dataset objects in `load_data()`
 - Define `compute_loss()` which takes a data iterator and
   computes the loss over it. This method is used for dev set perplexities.
 - Set `aux_loss` attribute for an additional loss term.
 - Define `forward()` which takes a dictionary with keys as data sources and
   returns the batch training loss. This is the method called from the `mainloop`
   during training.

Feel free to copy the methods from `NMT` if you do not need to modify
some of them.

#### Provided Models

Currently we only provide a **Conditional GRU NMT** [implementation](nmtpytorch/models/nmt.py)
with Bahdanau-style attention in decoder.

**NOTE**: We recommend limiting the number of tokens in the target vocabulary
by defining `max_trg_len` in the `[model]` section of your configuration file
to avoid GPU out of memory errors for very large vocabularies. This is caused
by the fact that the gradient computation for a batch with very long sequences
occupies a large amount of memory unless the loss layer is implemented differently.
# WMT18_MNMT
