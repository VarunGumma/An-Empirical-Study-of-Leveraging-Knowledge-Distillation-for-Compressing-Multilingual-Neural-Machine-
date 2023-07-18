INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"
import sys

from indicnlp import transliterate

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesDetokenizer

from indicnlp.tokenize import indic_detokenize
from indicnlp.transliterate import unicode_transliterate


def postprocess(
    input_size, lang, common_lang="hi", transliterate=False
):
    """
    parse fairseq interactive output, convert script back to native Indic script (in case of Indic languages) and detokenize.

    infname: fairseq log file
    outfname: output file of translation (sentences not translated contain the dummy string 'DUMMY_OUTPUT'
    input_size: expected number of output sentences
    lang: language
    """
    
    consolidated_testoutput = [(x, 0.0, "") for x in range(input_size)]

    temp_testoutput = list(
        map(
            lambda x: x.strip().split("\t"),
            filter(lambda x: x.startswith("H-"), list(sys.stdin)),
        )
    )
    temp_testoutput = list(
        map(lambda x: (int(x[0].split("-")[1]), float(x[1]), x[2]), temp_testoutput)
    )
    for sid, score, hyp in temp_testoutput:
        consolidated_testoutput[sid] = (sid, score, hyp)

    consolidated_testoutput = [x[2] for x in consolidated_testoutput]

    if lang == "en":
        en_detok = MosesDetokenizer(lang="en")
        for sent in consolidated_testoutput:
            print(en_detok.detokenize(sent.split(" ")))
    else:
        xliterator = unicode_transliterate.UnicodeIndicTransliterator()
        for sent in consolidated_testoutput:
            sent = xliterator.transliterate(sent, common_lang, lang) if transliterate else sent
            print(indic_detokenize.trivial_detokenize(sent, lang))



if __name__ == "__main__":

    input_size = int(sys.argv[1])
    lang = sys.argv[2]
    
    if len(sys.argv) == 3:
        transliterate = False
    elif len(sys.argv) == 4:
        transliterate = (sys.argv[3].lower() == "true")
    else:
        print(f"Invalid arguments: {sys.argv}")
        exit()

    postprocess(input_size, lang, common_lang="hi", transliterate=transliterate)
