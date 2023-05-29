import torch 
import os
from sys import argv

def func(path):
    for fname in os.listdir(path):
        temp_path = os.path.join(path, fname)
        if os.path.isdir(temp_path):
            func(temp_path)
        elif fname.endswith('.pt'):
            print(f"normalizing {temp_path}")
            ckpt = torch.load(temp_path)
            ckpt['cfg']['model']._arch = 'transformer'
            ckpt['cfg']['model']._name = 'transformer'
            torch.save(ckpt, temp_path)

if __name__ == '__main__':
    func(argv[1])