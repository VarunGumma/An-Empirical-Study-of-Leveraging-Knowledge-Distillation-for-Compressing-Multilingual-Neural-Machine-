## IndicDistillation
IndicDistillation is a repository that contains scripts for distilling [IndicTrans](https://github.com/AI4Bharat/indicTrans), a multilingual machine translation model for Indic languages.

### Setup
To get started, clone this repository and use it as the main experiment directory.

Install [fairseq](https://github.com/VarunGumma/fairseq) (with our mods) and [apex](https://github.com/NVIDIA/apex) (optional addon for faster training) in the environment you will be using with the following commands:
```
git clone https://github.com/VarunGumma/fairseq.git
cd fairseq
pip install --editable ./
```
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

Install [indic_nlp_library](https://github.com/anoopkunchukuttan/indic_nlp_library), [indic_nlp_resources](https://github.com/anoopkunchukuttan/indic_nlp_resources) and [subword-nmt](https://github.com/rsennrich/subword-nmt) in the repository using the following commands:

```
git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
cd indic_nlp_library
pip install --editable ./

git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git

git clone https://github.com/rsennrich/subword-nmt.git
cd subword-nmt
pip install --editable ./
```

### Optional libraries:
 - [ctranslate2](https://opennmt.net/CTranslate2/): For quantization and faster inference.
 - [xformers](https://github.com/facebookresearch/xformers): For faster and memory-efficient Transformers.


### Requirements
It is recommended to use ```Python 3.10+``` and the latest version of [PyTorch](https://pytorch.org/get-started/locally/) for the best results.

### License
This project is licensed under the MIT License.