INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"
import sys

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from collections import defaultdict

from tqdm import tqdm
from joblib import Parallel, delayed

from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate


en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()


def preprocess_line(line, normalizer, lang, transliterate=False):
    if lang == "en":
        return " ".join(
            en_tok.tokenize(en_normalizer.normalize(line.strip()), escape=False)
        )
    elif transliterate:
        # line = indic_detokenize.trivial_detokenize(line.strip(), lang)
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
            " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(line.strip()), lang
                )
            ),
            lang,
            "hi",
        ).replace(" ् ", "्")
    else:
        # we only need to transliterate for joint training
        return " ".join(
            indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), lang)
        )


def preprocess(infname, outfname, lang, transliterate=False):
    """
    Normalize, tokenize and script convert(for Indic)
    return number of sentences input file

    """

    n = 0
    num_lines = sum(1 for line in open(infname, "r"))
    if lang == "en":
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, None, lang)
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1

    else:
        normfactory = indic_normalize.IndicNormalizerFactory()
        normalizer = normfactory.get_normalizer(lang)
        # reading
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, normalizer, lang, transliterate)
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1
    return n


if __name__ == "__main__":

    infname = sys.argv[1]
    outfname = sys.argv[2]
    lang = sys.argv[3]

    if len(sys.argv) == 4:
        transliterate = False
    elif len(sys.argv) == 5:
        transliterate = sys.argv[4].lower() == "true"
    else:
        print(f"Invalid arguments: {sys.argv}")
        exit()
    print(preprocess(infname, outfname, lang, transliterate))
