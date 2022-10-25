## Indic-En
This repository contains scripts to distill the [IndicTrans](https://github.com/AI4Bharat/indicTrans) model. The final checkpoints of it can be found [here](https://drive.google.com/drive/u/1/folders/1bfF2m1UzzNe_M9SB6M60BfQeTsx4zq5j).

The project has been implemented using [fairseq](https://github.com/facebookresearch/fairseq) and our tweaks on top of it perform knowledge-distillation can be found at this [repo](https://github.com/VarunGumma/fairseq).

It is recommneded you run the distillation on at least 4 GPUs to get quick results. [apex](https://github.com/NVIDIA/apex) library is an optional addon for faster training.
