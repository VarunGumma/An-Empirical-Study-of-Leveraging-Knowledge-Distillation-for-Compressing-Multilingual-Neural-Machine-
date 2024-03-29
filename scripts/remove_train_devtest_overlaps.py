import os
import string
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

INDIC_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
# we will be testing the overlaps of training data with all these benchmarks

def read_lines(path):
    # if path doesnt exist, return empty list
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding='utf-8') as f:
        lines = [x.strip() for x in f]
    return lines


def pair_dedup_files(src_file, tgt_file):
    src_lines = read_lines(src_file)
    tgt_lines = read_lines(tgt_file)
    len_before = len(src_lines)

    src_dedupped, tgt_dedupped = pair_dedup_lists(src_lines, tgt_lines)

    len_after = len(src_dedupped)
    num_duplicates = len_before - len_after

    print(f"Dropped duplicate pairs in {src_file} Num duplicates -> {num_duplicates}")
    with open(src_file, 'w', encoding='utf-8') as out_src_file, \
         open(tgt_file, 'w', encoding='utf-8') as out_tgt_file:
        out_src_file.write('\n'.join(src_dedupped))
        out_tgt_file.write('\n'.join(tgt_dedupped))


def pair_dedup_lists(src_list, tgt_list):
    src_tgt = list(set(zip(src_list, tgt_list)))
    src_deduped, tgt_deduped = zip(*src_tgt)
    return src_deduped, tgt_deduped


def strip_and_normalize(line):
    # lowercase line, remove spaces and strip punctuation

    # one of the fastest way to add an exclusion list and remove that
    # list of characters from a string
    # https://towardsdatascience.com/how-to-efficiently-remove-punctuations-from-a-string-899ad4a059fb
    exclist = string.punctuation + "\u0964"
    table_ = str.maketrans("", "", exclist)

    line = line.replace(" ", "").lower()
    # dont use this method, it is painfully slow
    # line = "".join([i for i in line if i not in string.punctuation])
    line = line.translate(table_)
    return line


def expand_tupled_list(list_of_tuples):
    # convert list of tuples into two lists
    # https://stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
    # [(en, as), (as, bn), (bn, gu)] - > [en, as, bn], [as, bn, gu]
    list_a, list_b = map(list, zip(*list_of_tuples))
    return list_a, list_b


def get_src_tgt_lang_lists(many2many=False):
    if many2many is False:
        SRC_LANGS = ["en"]
        TGT_LANGS = INDIC_LANGS
    else:
        all_languages = INDIC_LANGS + ["en"]
        SRC_LANGS, TGT_LANGS = all_languages, all_languages

    return SRC_LANGS, TGT_LANGS


def normalize_and_gather_all_benchmarks(devtest_dir, many2many=False):
    # This is a dict of dict of lists
    # the first keys are for lang-pair, the second keys are for src/tgt
    # the values are the devtest lines.
    # so devtest_pairs_normalized[en-as][src] will store src(en lines)
    # so devtest_pairs_normalized[en-as][tgt] will store tgt(as lines)
    devtest_pairs_normalized = defaultdict(lambda: defaultdict(list))
    SRC_LANGS, TGT_LANGS = get_src_tgt_lang_lists(many2many)

    for dataset in os.listdir(devtest_dir):
        for src_lang in SRC_LANGS:
            for tgt_lang in TGT_LANGS:
                if src_lang == tgt_lang:
                    continue

                pair = f"{src_lang}-{tgt_lang}"
                if dataset == "wat2021-devtest":
                    # wat2021 dev and test sets have differnet folder structure
                    src_dev = read_lines(os.path.join(devtest_dir, dataset, f"dev.{src_lang}"))
                    tgt_dev = read_lines(os.path.join(devtest_dir, dataset, f"dev.{tgt_lang}"))
                    src_test = read_lines(os.path.join(devtest_dir, dataset, f"test.{src_lang}"))
                    tgt_test = read_lines(os.path.join(devtest_dir, dataset, f"test.{tgt_lang}"))
                else:
                    src_dev = read_lines(os.path.join(devtest_dir, dataset, pair, f"dev.{src_lang}"))
                    tgt_dev = read_lines(os.path.join(devtest_dir, dataset, pair, f"dev.{tgt_lang}"))
                    src_test = read_lines(os.path.join(devtest_dir, dataset, pair, f"test.{src_lang}"))
                    tgt_test = read_lines(os.path.join(devtest_dir, dataset, pair, f"test.{tgt_lang}"))

                # if the tgt_pair data doesnt exist for a particular test set,
                # it will be an empty list
                if tgt_test == [] or tgt_dev == []:
                    continue

                # combine both dev and test sets into one
                src_devtest = src_dev + src_test
                tgt_devtest = tgt_dev + tgt_test

                src_devtest = [strip_and_normalize(line) for line in src_devtest]
                tgt_devtest = [strip_and_normalize(line) for line in tgt_devtest]

                devtest_pairs_normalized[pair]["src"].extend(src_devtest)
                devtest_pairs_normalized[pair]["tgt"].extend(tgt_devtest)

    # dedup merged benchmark datasets
    for src_lang in SRC_LANGS:
        for tgt_lang in TGT_LANGS:
            if src_lang == tgt_lang:
                continue

            pair = f"{src_lang}-{tgt_lang}"
            src_devtest, tgt_devtest = (
                devtest_pairs_normalized[pair]["src"],
                devtest_pairs_normalized[pair]["tgt"],
            )
            # if the devtest data doesnt exist for the src-tgt pair then continue
            if src_devtest == [] or tgt_devtest == []:
                continue
            src_devtest, tgt_devtest = pair_dedup_lists(src_devtest, tgt_devtest)
            (
                devtest_pairs_normalized[pair]["src"],
                devtest_pairs_normalized[pair]["tgt"],
            ) = (
                src_devtest,
                tgt_devtest,
            )

    return devtest_pairs_normalized


def remove_train_devtest_overlaps(in_dir, devtest_dir, out_dir, many2many=False):

    devtest_pairs_normalized = normalize_and_gather_all_benchmarks(devtest_dir, many2many)

    SRC_LANGS, TGT_LANGS = get_src_tgt_lang_lists(many2many)

    if not many2many:
        all_src_sentences_normalized = []
        for key in devtest_pairs_normalized:
            all_src_sentences_normalized.extend(devtest_pairs_normalized[key]["src"])
        # remove all duplicates. Now this contains all the normalized
        # english sentences in all test benchmarks across all lang pair
        all_src_sentences_normalized = list(set(all_src_sentences_normalized))
    else:
        all_src_sentences_normalized = None

    src_overlaps = []
    tgt_overlaps = []
    for src_lang in SRC_LANGS:
        for tgt_lang in TGT_LANGS:
            if src_lang == tgt_lang:
                continue
            new_src_train = []
            new_tgt_train = []

            pair = f"{src_lang}-{tgt_lang}"
            src_train = read_lines(os.path.join(in_dir, pair, f"train.{src_lang}"))
            tgt_train = read_lines(os.path.join(in_dir, pair, f"train.{tgt_lang}"))

            len_before = len(src_train)
            if len_before == 0:
                continue

            src_train_normalized = [strip_and_normalize(line) for line in src_train]
            tgt_train_normalized = [strip_and_normalize(line) for line in tgt_train]

            if all_src_sentences_normalized:
                src_devtest_normalized = all_src_sentences_normalized
            else:
                src_devtest_normalized = devtest_pairs_normalized[pair]["src"]

            tgt_devtest_normalized = devtest_pairs_normalized[pair]["tgt"]

            # compute all src and tgt super strict overlaps for a lang pair
            overlaps = set(src_train_normalized) & set(src_devtest_normalized)
            src_overlaps.extend(list(overlaps))

            overlaps = set(tgt_train_normalized) & set(tgt_devtest_normalized)
            tgt_overlaps.extend(list(overlaps))
            # dictionaries offer o(1) lookup
            src_overlaps_dict = {}
            tgt_overlaps_dict = {}
            for line in src_overlaps:
                src_overlaps_dict[line] = True
            for line in tgt_overlaps:
                tgt_overlaps_dict[line] = True

            # loop to remove the ovelapped data
            for (idx, (src_line_norm, tgt_line_norm)) in tqdm(enumerate(zip(src_train_normalized, tgt_train_normalized)), total=len_before):
                if not src_overlaps_dict.get(src_line_norm, False) and \
                   not tgt_overlaps_dict.get(tgt_line_norm, False):
                    new_src_train.append(src_train[idx])
                    new_tgt_train.append(tgt_train[idx])

            len_after = len(new_src_train)
            print(f"Detected overlaps between train and devtest for {pair} is {len_before - len_after}")

            os.makedirs(os.path.join(out_dir, pair), exist_ok=True)
            print(f"saving new files at {os.path.join(out_dir, pair)}")

            with open(os.path.join(out_dir, pair, f"train.{src_lang}"), "w", encoding='utf-8') as out_src_file, \
                 open(os.path.join(out_dir, pair, f"train.{tgt_lang}"), "w", encoding='utf-8') as out_tgt_file:
                out_src_file.write('\n'.join(new_src_train))
                out_tgt_file.write('\n'.join(new_tgt_train))



if __name__ == "__main__":
    parser = ArgumentParser(description="remove train_devtest overlaps")
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--devtest-dir", type=str, required=True)
    parser.add_argument("--many2many", action="store_true")
    parser.add_argument("--languages-list", type=str, default="as+bn+gu+hi+kn+ml+mr+or+pa+ta+te", required=True)
    args = parser.parse_args()

    INDIC_LANGS = args.languages_list.split('+')

    remove_train_devtest_overlaps(
        in_dir=args.in_dir, 
        devtest_dir=args.devtest_dir, 
        many2many=args.many2many, 
        out_dir=args.out_dir
    )
