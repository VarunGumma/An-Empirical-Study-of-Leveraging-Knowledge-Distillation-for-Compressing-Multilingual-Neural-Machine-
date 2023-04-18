import torch 
import os
from sys import argv

def func(path):
    for fname in os.listdir(path):
        if os.path.isdir(fname):
            func(f"{path}/{fname}")
        elif fname.endswith('.pt'):
            model_path = f"{path}/{fname}"
            print(f"normalizing {model_path}")
            ckpt = torch.load(model_path)
            ckpt['cfg']['model']._arch = 'transformer'
            ckpt['cfg']['model']._name = 'transformer'
            torch.save(ckpt, model_path)

func(argv[1])