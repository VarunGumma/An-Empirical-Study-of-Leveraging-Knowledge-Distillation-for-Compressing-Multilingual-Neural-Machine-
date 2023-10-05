## IndicDistillation
IndicDistillation is a repository that contains scripts for distilling [IndicTrans](https://github.com/AI4Bharat/indicTrans), a multilingual machine translation model for Indic languages, and also part of our work [An Empirical Study of Leveraging Knowledge Distillation for Compressing Multilingual Neural Machine Translation Models](https://aclanthology.org/2023.eamt-1.11/)

### Setup
To get started, clone this repository and use it as the main experiment directory.

Install [fairseq](https://github.com/VarunGumma/fairseq) (with our mods) and [apex](https://github.com/NVIDIA/apex) (optional addon for faster training) in the environment you will be using with the following commands:
```
git clone https://github.com/VarunGumma/fairseq.git
cd fairseq
pip install -e ./
```
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

Install [indic_nlp_library](https://github.com/VarunGumma/indic_nlp_library), and [subword-nmt](https://github.com/rsennrich/subword-nmt) in the repository using the following commands:

```
git clone https://github.com/VarunGumma/indic_nlp_library.git
cd indic_nlp_library
pip install -e ./
pip install subword-nmt
```

### Optional libraries:
 - [ctranslate2](https://opennmt.net/CTranslate2/): For quantization and faster inference.
 - [xformers](https://github.com/facebookresearch/xformers): For faster and memory-efficient Transformers.


### Requirements
It is recommended to use `Python 3.10+` and the latest version of [PyTorch](https://pytorch.org/get-started/locally/) for the best results.

## License
This project is licensed under the MIT License.

## Citation
If you use these code or models, please cite our work:
```
@inproceedings{gumma-etal-2023-empirical,
    title = "An Empirical Study of Leveraging Knowledge Distillation for Compressing Multilingual Neural Machine Translation Models",
    author = "Gumma, Varun  and
      Dabre, Raj  and
      Kumar, Pratyush",
    booktitle = "Proceedings of the 24th Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.eamt-1.11",
    pages = "103--114",
    abstract = "Knowledge distillation (KD) is a well-known method for compressing neural models. However, works focusing on distilling knowledge from large multilingual neural machine translation (MNMT) models into smaller ones are practically nonexistent, despite the popularity and superiority of MNMT. This paper bridges this gap by presenting an empirical investigation of knowledge distillation for compressing MNMT models. We take Indic to English translation as a case study and demonstrate that commonly used language-agnostic and language-aware KD approaches yield models that are 4-5x smaller but also suffer from performance drops of up to 3.5 BLEU. To mitigate this, we then experiment with design considerations such as shallower versus deeper models, heavy parameter sharing, multistage training, and adapters. We observe that deeper compact models tend to be as good as shallower non-compact ones and that fine-tuning a distilled model on a high-quality subset slightly boosts translation quality. Overall, we conclude that compressing MNMT models via KD is challenging, indicating immense scope for further research.",
}
```
