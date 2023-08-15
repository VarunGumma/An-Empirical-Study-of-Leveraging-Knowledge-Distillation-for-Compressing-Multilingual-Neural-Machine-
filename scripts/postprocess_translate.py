INDIC_NLP_LIB_HOME = "/home/varun/indic_nlp_library"
INDIC_NLP_RESOURCES = "/home/varun/indic_nlp_resources"
import sys

from indicnlp import transliterate

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesDetokenizer
from joblib import Parallel, delayed

from indicnlp.tokenize import indic_detokenize
from indicnlp.transliterate import unicode_transliterate

en_detok = MosesDetokenizer(lang="en")
xliterator = unicode_transliterate.UnicodeIndicTransliterator()


def postprocess(
    infname,
    outfname,
    input_size, 
    lang, 
    transliterate=False
):
    """
    parse fairseq interactive output, convert script back to native Indic script (in case of Indic languages) and detokenize.

    infname: fairseq log file
    outfname: output file of translation (sentences not translated contain the dummy string 'DUMMY_OUTPUT'
    input_size: expected number of output sentences
    lang: language
    """
    
    consolidated_testoutput = [(x, 0.0, "") for x in range(input_size)]

    with open(infname, "r", encoding="utf-8") as infile:
        temp_testoutput = list(
            map(
                lambda x: x.strip().split("\t"),
                filter(lambda x: x.startswith("H-"), infile),
            )
        )

    temp_testoutput = list(
        map(
            lambda x: (int(x[0].split("-")[1]), float(x[1]), x[2]), temp_testoutput
        )
    )

    for sid, score, hyp in temp_testoutput:
        consolidated_testoutput[sid] = (sid, score, hyp)

    consolidated_testoutput = [x[2] for x in consolidated_testoutput]

    if lang == "en":
        consolidated_testoutput = Parallel(n_jobs=-1)([delayed(en_detok.detokenize)(x.split(' ')) for x in consolidated_testoutput])
    else:
        f = lambda sent: xliterator.transliterate(sent, "hi", lang) if transliterate else sent
        consolidated_testoutput = Parallel(n_jobs=-1)([delayed(indic_detokenize.trivial_detokenize)(f(x), lang) for x in consolidated_testoutput])

    with open(outfname, "w", encoding="utf-8") as outfile:
        outfile.write('\n'.join(consolidated_testoutput))


if __name__ == "__main__":
    infname = sys.argv[1]
    outfname = sys.argv[2]
    input_size = int(sys.argv[3])
    lang = sys.argv[4]
    transliterate = sys.argv[5]

    postprocess(infname, outfname, input_size, lang, transliterate)
