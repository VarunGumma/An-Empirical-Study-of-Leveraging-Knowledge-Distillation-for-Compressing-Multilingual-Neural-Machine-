import pandas as pd
from os import listdir
from subprocess import check_output

base_path = "comet"
comet_model = "wmt21-comet-qe-mqm"
all_results = []

print(f"using {comet_model}")

for model in ['base', 'it']:
    for dirname in sorted(listdir(base_path)):
        for lang_pair in sorted(listdir(f"{base_path}/{dirname}")):
            full_path = f"{base_path}/{dirname}/{lang_pair}"
            tgt = lang_pair.split('-')[-1]
            
            # cmd = f"comet-score -s {full_path}/test.{tgt} -r {full_path}/test.en -t {full_path}/test_{model}.en --quiet --num_workers 32 --batch_size 256 --seed_everything 2023 --model {comet_model}"
            cmd = f"comet-score -s {full_path}/test.{tgt} -t {full_path}/test_{model}.en --quiet --num_workers 32 --batch_size 256 --seed_everything 2023 --model {comet_model}"

            out = check_output(cmd, shell=True, text=True).strip()
            _path, _score = out.split('\t')
            score = float(_score.split(': ')[-1])
            benchmark = f"{_path.split('/')[1]} - {tgt}"

            all_results.append([model, benchmark, score])
            print([model, benchmark, score])

df = pd.DataFrame(all_results, columns=['model', 'benchmark', 'score'])
df = df.pivot(*df).rename_axis(columns=None).reset_index()
df.to_csv(f"{comet_model}-scores.csv")