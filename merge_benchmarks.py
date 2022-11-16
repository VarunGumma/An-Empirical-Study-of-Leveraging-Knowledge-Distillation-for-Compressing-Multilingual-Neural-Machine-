from os import system, listdir
from sys import argv

devtest_folders = ["flores101_dataset", "pmi", "ufal-ta", "wat2020-devtest", "wat2021-devtest", "wmt-news"]
languages = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
BASE_PATH = argv[1]

system(f"mkdir {BASE_PATH}/all")

for lang in languages:
    lang_merged_dev = []
    lang_merged_test = []
    en_merged_dev = []
    en_merged_test = []
    lang_pair  = f"en-{lang}"
    system(f"mkdir {BASE_PATH}/all/{lang_pair}")

    for devtest_folder in devtest_folders:
        if lang_pair in listdir(f"{BASE_PATH}/{devtest_folder}"):
            if f"dev.{lang}" in listdir(f"{BASE_PATH}/{devtest_folder}/{lang_pair}"):
                with open(f"{BASE_PATH}/{devtest_folder}/{lang_pair}/dev.{lang}", 'r') as f:
                    lang_merged_dev.extend(f.readlines())
            if "dev.en" in listdir(f"{BASE_PATH}/{devtest_folder}/{lang_pair}"):
                with open(f"{BASE_PATH}/{devtest_folder}/{lang_pair}/dev.en", 'r') as f:
                    en_merged_dev.extend(f.readlines())
            if f"test.{lang}" in listdir(f"{BASE_PATH}/{devtest_folder}/{lang_pair}"):
                with open(f"{BASE_PATH}/{devtest_folder}/{lang_pair}/test.{lang}", 'r') as f:
                    lang_merged_test.extend(f.readlines())
            if "test.en" in listdir(f"{BASE_PATH}/{devtest_folder}/{lang_pair}"):
                with open(f"{BASE_PATH}/{devtest_folder}/{lang_pair}/test.en", 'r') as f:
                    en_merged_test.extend(f.readlines())
                
    with open(f"{BASE_PATH}/all/{lang_pair}/dev.{lang}", 'w') as f1, \
         open(f"{BASE_PATH}/all/{lang_pair}/dev.en", 'w') as f2, \
         open(f"{BASE_PATH}/all/{lang_pair}/test.{lang}", 'w') as f3, \
         open(f"{BASE_PATH}/all/{lang_pair}/test.en", 'w') as f4:
        f1.write('\n'.join(lang_merged_dev))
        f2.write('\n'.join(en_merged_dev))
        f3.write('\n'.join(lang_merged_test))
        f4.write('\n'.join(en_merged_test))