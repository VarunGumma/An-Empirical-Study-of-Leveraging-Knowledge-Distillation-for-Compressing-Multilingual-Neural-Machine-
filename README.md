## Indic-En
This repository contains scripts to distil the [IndicTrans](https://github.com/AI4Bharat/indicTrans) ```XX-En``` model. 

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

The samanantar binarized data can be found [here](https://drive.google.com/file/d/1ZKyLzN4mUzG5Yy8GdPj02HXDD7k53tj2/view?usp=share_link). Unzip it in the ```Indic-En``` directory and you will get two folders, namely ```indic-en-exp``` and ```indic-en``` which contain the binarized distilled and original Samanantar data respectively. The ```indic-en``` directory also contains the best checkpoint of IndicTrans. 

Before training, create a ```checkpoints``` folder in the ```Indic-En``` directory. This will hold all checkpoints of the all the trained models.


### Training
Each model being trained is programmed to use 8 GPUs and 16 CPUs. In case required you can change these in the scripts by modifying the ```--distributed-world-size``` and ```--num-workers``` arguments. After modification make sure that the global batch size ```(distributed_world_size * max_tokens * update_freq)``` is 64K. Since ```fp16``` mixed-precision is prone to underflow during distillation or training deeper models, we recommend using ```fp32``` itself.

All the files which contain the term ```distil``` perform ```word + seq level``` distillation (training a smaller model using distilled data and teacher distibution signal). 

Files which contain the term ```train``` perform ```seq level``` distillation (training a smaller model using distilled data only)

If a file contains the term ```4x```, the dimension of the model which will be trained is 1536, which is same as IndicTrans, else it is a ```1x``` model with dimension 512.

we use ```--activation-fn=gelu```, ```--encoder-normalize-before```, ```--decoder-normalize-before``` and ```--layernorm-embedding``` arguments to stablize the training of all the distilled models. Note that these tweaks are not present in the original IndicTrans model.


### Evaluation
Once all the required models have been trained, their checkpoints will be available in the ```checkpoints``` directory. You can use the following commands to generate a ```csv``` of benchmark scores.
```
cd indicTrans
bash evaluate_benchmarks.sh <ckpt-folder-1> <ckpt-folder-2> <ckpt-folder-3> <ckpt-folder-3> ... 
```
Once it has executed, all the required results will be available in the ```results/benchmark_scores.csv``` file. 

Make sure the ```results``` folder always contains the ```indicTrans.txt``` file which contains the standard benchmark scores of IndicTrans model (as given [here](https://github.com/AI4Bharat/indicTrans))
