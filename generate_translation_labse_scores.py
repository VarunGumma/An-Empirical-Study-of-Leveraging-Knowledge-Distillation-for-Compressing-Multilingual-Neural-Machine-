import argparse
import logging
import torch
from os import listdir
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.DEBUG)

@torch.no_grad()
def main(args):
    model = SentenceTransformer("sentence-transformers/LaBSE")
    logging.info(" - labse model loaded successfully from sentence-transformers")

    for dirname in listdir(args.folder):
        tgt_lang, src_lang = dirname.split("-")
        logging.info(f" - working on {src_lang}")
        with open(f"{args.folder}/{dirname}/train.{src_lang}", "r") as f1, \
             open(f"{args.folder}/{dirname}/train.{tgt_lang}", "r") as f2 :
            src_sents, tgt_sents = f1.readlines(), f2.readlines()

        cos_sim, idx, L = [], 0, len(src_sents)

        while idx < L:
            # calculate the encoded representation for both source and target sentences
            src_embeds = model.encode(src_sents[idx : idx + args.batch_size], batch_size=1024, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
            tgt_embeds = model.encode(tgt_sents[idx : idx + args.batch_size], batch_size=1024, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
            # calculate the cosine similarity
            cos_sim.extend(torch.mul(src_embeds, tgt_embeds).sum(1).detach().tolist())
            logging.info(f" - scores calculated for {len(cos_sim)} samples")
            idx = min(L, idx + args.batch_size)
    
        with open(f"{args.folder}/{dirname}/scores.txt", "w") as f:
            f.write('\n'.join(map(str, cos_sim)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("cosine similarity calculation")
    parser.add_argument("-f", "--folder", help="location of the root directory")
    parser.add_argument("-b", "--batch-size", type=int, default=65536, help="batch size for the model")
    args = parser.parse_args()
    main(args)
