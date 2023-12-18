import sys
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate
from sacremoses import MosesPunctNormalizer, MosesTokenizer

loader.load()
en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()
normfactory = indic_normalize.IndicNormalizerFactory()


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
    normalizer = normfactory.get_normalizer(lang) if lang != 'en' else None

    for line in sys.stdin:
        print(preprocess_line(line, normalizer, lang, transliterate))


if __name__ == "__main__":

    lang = sys.argv[1]
    transliterate = sys.argv[2]

    transliterate = transliterate.lower() == "true"
    
    preprocess(lang=lang, transliterate=transliterate)
