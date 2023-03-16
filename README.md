## Indic-En
This repository contains scripts to distill [IndicTrans](https://github.com/AI4Bharat/indicTrans)

### Setup
Clone this repository, and it will act as the main experiment directory from now on.

Install [fairseq](https://github.com/VarunGumma/fairseq) (with our mods) and [apex](https://github.com/NVIDIA/apex) (optional addon fort faster training) in the environment you will be using with the following commands:
```
cd fairseq
pip install --editable ./
```
```
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

It is recommended you use ```Python 3.10+``` and the latest version of [Pytorch](https://pytorch.org/get-started/locally/) for best results.
