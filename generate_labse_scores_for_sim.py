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

    for dirname in sorted(listdir(args.original)):
        logging.info(f" - working on {dirname}")
        lang = dirname.split("-")[1] if args.language == "indic" else "en"
        logging.info(f" - reading {lang} translations of original and distilled data")
        with open(f"{args.original}/{dirname}/train.{lang}", "r") as f1, \
             open(f"{args.distilled}/{dirname}/train.{lang}", "r") as f2 :
            og, distil = f1.readlines(), f2.readlines()

        cos_sim, idx, L = [], 0, len(og)

        while idx < L:
            # calculate the encoded representation for both source and target sentences
            og_embeds = model.encode(og[idx : idx + args.batch_size], batch_size=1024, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
            distil_embed = model.encode(distil[idx : idx + args.batch_size], batch_size=1024, device='cuda', normalize_embeddings=True, convert_to_tensor=True)
            # calculate the cosine similarity
            cos_sim.extend(torch.mul(og_embeds, distil_embed).sum(1).detach().tolist())
            logging.info(f" - scores calculated for {len(cos_sim)} samples")
            idx = min(L, idx + args.batch_size)
            torch.cuda.empty_cache()
    
        with open(f"{args.original}/{dirname}/og_distilled_sim_scores.txt", "w") as f:
            f.write('\n'.join(map(str, cos_sim)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("cosine similarity calculation")
    parser.add_argument("-o", "--original", type=str, help="location of the original data")
    parser.add_argument("-d", "--distilled", type=str, help="location of distilled data")
    parser.add_argument("-l", "--language", type=str, help="language to be compared with its distilled version")
    parser.add_argument("-b", "--batch-size", type=int, help="batch size for the model")
    args = parser.parse_args()
    main(args)
