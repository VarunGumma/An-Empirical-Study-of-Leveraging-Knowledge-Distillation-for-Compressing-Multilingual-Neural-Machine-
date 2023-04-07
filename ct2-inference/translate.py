from os import listdir, system, makedirs
from sys import argv
from engine import Model
import argparse

parser = argparse.ArgumentParser(description='Evaluate a model using CT2')
parser.add_argument('--devtest-dir', '-d', help='devtest directory path; preferably flores')
parser.add_argument('--exp-dir', '-e', help='experiment directory path')
parser.add_argument('--model-name', '-m', help='model name; this ct2-converted model should be present in the experiment directory')
parser.add_argument('--source-lang', '-s', default='indic', help='source language')
parser.add_argument('--target-lang', '-t', default='en', help='target language')
parser.add_argument('--beam-size', default=5, type=int, help='beam size for decoding')
parser.add_argument('--batch-size', default=1024, type=int, help='batch size for inference')
parser.add_argument('--device', default='cuda', help='device for inference; cpu/cuda')

args = parser.parse_args()
devtest_dir = args.devtest_dir
exp_dir = args.exp_dir
model_name = args.model_name
src = args.source_lang
tgt = args.target_lang
beam_size = args.beam_size
batch_size = args.batch_size
device = args.device

model = Model(
    exp_dir=exp_dir, 
    model_name=model_name,
    device=device
)

system("rm -rf ../results/*")

for lang_pair in sorted(listdir(devtest_dir)):
    if src == 'en':
        tgt = lang_pair.split('-')[1]
    else:
        src = lang_pair.split('-')[1]

    with open(f"{devtest_dir}/{lang_pair}/test.{src}", 'r', encoding='utf-8') as f:
        src_sents = [line.strip() for line in f]

    translated_sents = model.batch_translate(
        src_sents,
        src_lang=src,
        tgt_lang=tgt,
        beam_size=int(beam_size),
        max_batch_size=int(batch_size)
    )

    with open(f"{devtest_dir}/{lang_pair}/outfile.{tgt}", 'w', encoding='utf-8') as f:
        f.write('\n'.join(translated_sents))

    makedirs(f"../results/{model_name}", exist_ok=True)
    system(f"bash ../compute_bleu.sh {devtest_dir}/{lang_pair}/outfile.{tgt} {devtest_dir}/{lang_pair}/test.{tgt} {src} {tgt} > ../results/{model_name}/{lang_pair}.json")

system(f"python3 ../json_to_csv.py ../results {model_name}")