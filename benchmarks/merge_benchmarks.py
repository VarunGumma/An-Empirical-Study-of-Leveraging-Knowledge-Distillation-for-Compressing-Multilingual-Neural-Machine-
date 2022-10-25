from os import system, listdir

devtest_folders = ["flores101_dataset", "pmi", "ufal-ta", "wat2020-devtest", "wat2021-devtest", "wmt-news"]
languages = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

system("mkdir all")
for lang in languages:
    system(f"mkdir ./all/en-{lang}")

for lang in languages:
    lang_merged_dev = []
    lang_merged_test = []
    en_merged_dev = []
    en_merged_test = []
    lang_folder_name  = f"en-{lang}"

    for devtest_folder in devtest_folders:
        if lang_folder_name in listdir(f"./{devtest_folder}"):
            if f"dev.{lang}" in listdir(f"./{devtest_folder}/{lang_folder_name}"):
                with open(f"./{devtest_folder}/{lang_folder_name}/dev.{lang}", 'r') as f:
                    lang_merged_dev.extend(f.readlines())
            if "dev.en" in listdir(f"./{devtest_folder}/{lang_folder_name}"):
                with open(f"./{devtest_folder}/{lang_folder_name}/dev.en", 'r') as f:
                    en_merged_dev.extend(f.readlines())
            if f"test.{lang}" in listdir(f"./{devtest_folder}/{lang_folder_name}"):
                with open(f"./{devtest_folder}/{lang_folder_name}/test.{lang}", 'r') as f:
                    lang_merged_test.extend(f.readlines())
            if "test.en" in listdir(f"./{devtest_folder}/{lang_folder_name}"):
                with open(f"./{devtest_folder}/{lang_folder_name}/test.en", 'r') as f:
                    en_merged_test.extend(f.readlines())
                
    with open(f"./all/{lang_folder_name}/dev.{lang}", 'w') as f:
        f.write('\n'.join(lang_merged_dev))
    with open(f"./all/{lang_folder_name}/dev.en", 'w') as f:
        f.write('\n'.join(en_merged_dev))
    with open(f"./all/{lang_folder_name}/test.{lang}", 'w') as f:
        f.write('\n'.join(lang_merged_test))
    with open(f"./all/{lang_folder_name}/test.en", 'w') as f:
        f.write('\n'.join(en_merged_test))