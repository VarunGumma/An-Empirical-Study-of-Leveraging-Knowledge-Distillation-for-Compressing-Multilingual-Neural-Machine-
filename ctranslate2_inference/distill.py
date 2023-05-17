from os import listdir, system, makedirs
from sys import argv
from engine import Model
import argparse
import shutil
import tqdm
from time import sleep

parser = argparse.ArgumentParser(description='Evaluate a model using CT2')
parser.add_argument('--src-dir', help="source directory")
parser.add_argument('--dest-dir', help="destination directory")
parser.add_argument('--exp-dir', '-e', help="experiment directory")
parser.add_argument('--model-name', '-m', help='model name; this ct2-converted model should be present in the experiment directory')
parser.add_argument('--source-lang', '-s', default='indic', help='source language')
parser.add_argument('--target-lang', '-t', default='en', help='target language')
parser.add_argument('--beam-size', default=5, type=int, help='beam size for decoding')
parser.add_argument('--batch-size', default=1024, type=int, help='batch size for inference')
parser.add_argument('--device', default='cuda', help='device for inference; cpu/cuda')
parser.add_argument('--dir-num', default=0, type=int, help="temporary flag, will be removed")

args = parser.parse_args()
src_dir = args.src_dir
dest_dir = args.dest_dir
exp_dir = args.exp_dir
model_name = args.model_name
src = args.source_lang
tgt = args.target_lang
beam_size = args.beam_size
batch_size = args.batch_size
device = args.device
n = args.dir_num

def batch_generator(x, bs=32):
    n = len(x)
    for i in range(0, n, bs):
        j = min(n, i+bs)
        yield x[i : j]


model = Model(
    exp_dir=exp_dir, 
    model_name=model_name,
    device=device
)

for lang_pair in sorted(listdir(src_dir))[n:n+1]:
    if src == 'en':
        tgt = lang_pair.split('-')[1]
    else:
        src = lang_pair.split('-')[1]

    makedirs(f"{dest_dir}/{lang_pair}", exist_ok=True)
    shutil.copy(f"{src_dir}/{lang_pair}/train.{src}", f"{dest_dir}/{lang_pair}")

    with open(f"{dest_dir}/{lang_pair}/train.{src}", 'r', encoding='utf-8') as f:
        src_sents = [line.strip() for line in f]
        batched_src_sents = batch_generator(src_sents, bs=batch_size)

    translated_sents = []
    print(f"starting translations for {lang_pair}")
    for batch in batched_src_sents:
        batch_translated_sents = model.batch_translate(
            batch,
            src_lang=src,
            tgt_lang=tgt,
            beam_size=int(beam_size),
            max_batch_size=len(batch)
        )

        translated_sents.extend(batch_translated_sents)

    print(f"reached here {len(translated_sents)}")

    with open(f"{dest_dir}/{lang_pair}/train.{tgt}", 'w', encoding='utf-8') as f:
        f.write('\n'.join(translated_sents))

    print(f"completed translations for {lang_pair}")
