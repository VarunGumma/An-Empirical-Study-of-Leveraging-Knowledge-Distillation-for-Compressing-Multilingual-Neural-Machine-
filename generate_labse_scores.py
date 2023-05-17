import argparse
import logging
import torch
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.DEBUG)

@torch.no_grad()
def main(args):
    model = SentenceTransformer("sentence-transformers/LaBSE")
    logging.info(" - labse model loaded successfully from sentence-transformers")

    tgt_lang, src_lang = args.lang_pair.split("-")
    logging.info(f" - working on {src_lang}-{tgt_lang}")
    with open(f"{args.folder}/{args.lang_pair}/train.{src_lang}", "r", encoding='utf-8') as f1, \
         open(f"{args.folder}/{args.lang_pair}/train.{tgt_lang}", "r") as f2 :
        src_sents = [x.strip() for x in f1]
        tgt_sents = [x.strip() for x in f2]

    cos_sim, idx, L = [], 0, len(src_sents)

    while idx < L:
        # calculate the encoded representation for both source and target sentences
        src_embeds = model.encode(
            src_sents[idx : idx + args.batch_size], 
            batch_size=2048, 
            device='cuda', 
            normalize_embeddings=True, 
            convert_to_tensor=True
        )
        tgt_embeds = model.encode(
            tgt_sents[idx : idx + args.batch_size], 
            batch_size=2048, 
            device='cuda', 
            normalize_embeddings=True, 
            convert_to_tensor=True
        )
        # calculate the cosine similarity
        cos_sim.extend(torch.mul(src_embeds, tgt_embeds).sum(1).detach().tolist())
        logging.info(f" - scores calculated for {len(cos_sim)} samples")
        idx = min(L, idx + args.batch_size)

    assert len(cos_sim) == len(src_sents), "size mismatch between LaBSE scores and source sentences."

    with open(f"{args.folder}/{args.lang_pair}/labse.txt", "w") as f:
        f.write('\n'.join(map(str, cos_sim)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("cosine similarity calculation")
    parser.add_argument("-f", "--folder", help="location of the root directory")
    parser.add_argument('-l', '--lang-pair', help="language pair")
    parser.add_argument("-b", "--batch-size", type=int, default=1048576, help="batch size for the model")
    args = parser.parse_args()
    main(args)
