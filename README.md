# Junction Tree Variational Autoencoder for Molecular Graph Generation

<img src="https://github.com/wengong-jin/icml18-jtnn/blob/master/paradigm.png" width="400">

Implementation of Junction Tree Variational Autoencoder from Jin et al. [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)

This is a fork to port the model onto Python 3 and some library.

Currently only VAE training is ported.

You need to initialize the submodule by the following:

```
git submodule init
git submodule update
```

# Checklist

- [x] VAE training
  * Run `PYTHONPATH=$PWD/_dgl/python:$PWD:$PYTHONPATH python3 molvae/vaetrain.py -t data/train.txt -v data/vocab.txt -s . -b 10`
  * Or `PYTHONPATH=$PWD/_dgl/python:$PWD:$PYTHONPATH python3 molvae/vaetrain_dgl.py -t data/train.txt -v data/vocab.txt -s . -b 10`
- [ ] VAE decoding during inference
- [ ] Property prediction
- [ ] CUDA support
- [x] Profiling
  * To enable profiling, set the environment variable `PROFILE` to 1, and set `PROFILE_OUTPUT` to the destination of line profiler
    dump (leave it blank for standard output)

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2017.09)
* Python (version >= 2.7)
* PyTorch (version >= 0.2)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

# Quick Start
This repository contains the following directories:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training our VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.

# Contact
Quan (Andy) Gan (qg323@nyu.edu)
