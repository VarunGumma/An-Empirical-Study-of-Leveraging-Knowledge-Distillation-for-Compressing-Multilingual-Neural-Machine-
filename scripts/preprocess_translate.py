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

from tqdm import tqdm
from joblib import Parallel, delayed

from indicnlp.tokenize import indic_tokenize
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


def preprocess(lang, transliterate=False):
    """
    Normalize, tokenize and script convert(for Indic)
    return number of sentences input file

    """
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(lang) if lang != 'en' else None

    for line in sys.stdin:
        print(preprocess_line(line, normalizer, lang, transliterate))


if __name__ == "__main__":

    lang = sys.argv[1]
    transliterate = sys.argv[2]

    transliterate = transliterate.lower() == "true"
    
    preprocess(lang=lang, transliterate=transliterate)
