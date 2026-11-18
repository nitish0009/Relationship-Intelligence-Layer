#!/usr/bin/env python3
"""Train the ORIGINAL CARE-GNN (Dou et al., CIKM 2020) or PC-GNN (Liu et al., WWW 2021) code
on an exported graph and write val/test fraud probabilities.

Runs in its own process with ONLY the original repository on sys.path (PC-GNN's package is
called ``src``, which would clash with ours). Hyper-parameters are the authors' published
defaults (CARE-GNN train.py defaults; PC-GNN config/pcgnn_yelpchi.yml). The only changes are
(a) our time-respecting graph/features as input and (b) model selection by validation PR-AUC.

Usage: run_original_gnn.py {care,pcgnn} graph.pkl out.npz [--seed S] [--epochs E]
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "third_party")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["care", "pcgnn"])
    ap.add_argument("graph")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=72)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--no-row-norm", action="store_true",
                    help="diagnostic: skip the authors' row-sum feature normalisation (designed for bag-of-words)")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    import scipy.sparse as sp
    from sklearn.metrics import average_precision_score

    repo = os.path.join(ROOT, "CARE-GNN" if a.kind == "care" else "PC-GNN")
    if not os.path.isdir(repo):
        sys.exit(f"{repo} missing: run scripts/fetch_third_party.sh")
    sys.path.insert(0, repo)
    if a.kind == "care":
        from layers import InterAgg, IntraAgg
        from model import OneLayerCARE
        from utils import normalize, pos_neg_split, undersample
        cfg = dict(lr=0.01, weight_decay=1e-3, emb=64, epochs=31, batch=1024, lambda_1=2.0, step=2e-2)
    else:
        from src.layers import InterAgg, IntraAgg
        from src.model import PCALayer
        from src.utils import normalize, pick_step, pos_neg_split
        cfg = dict(lr=0.01, weight_decay=1e-3, emb=64, epochs=51, batch=1024, alpha=2.0, rho=0.5)
    if a.epochs:
        cfg["epochs"] = a.epochs

    with open(a.graph, "rb") as f:
        g = pickle.load(f)
    np.random.seed(a.seed); random.seed(a.seed); torch.manual_seed(a.seed)
    labels = g["labels"].astype(np.int64)
    feat = g["features"].astype(np.float32) if a.no_row_norm else \
        np.asarray(normalize(sp.csr_matrix(g["features"])).todense(), dtype=np.float32)
    features = nn.Embedding(feat.shape[0], feat.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat), requires_grad=False)
    idx_train = list(map(int, g["idx_train"]))
    y_train = labels[idx_train]
    train_pos, train_neg = pos_neg_split(idx_train, y_train)
    adj = g["adj"]  # list of 3 dicts node -> set(neighbours incl. self); neighbours strictly earlier

    if a.kind == "care":
        intras = [IntraAgg(features, feat.shape[1], cuda=False) for _ in range(3)]
        inter = InterAgg(features, feat.shape[1], cfg["emb"], adj, intras, inter="GNN", step_size=cfg["step"], cuda=False)
        model = OneLayerCARE(2, inter, cfg["lambda_1"])
    else:
        intras = [IntraAgg(features, feat.shape[1], cfg["emb"], train_pos, cfg["rho"], cuda=False) for _ in range(3)]
        inter = InterAgg(features, feat.shape[1], cfg["emb"], train_pos, adj, intras, inter="GNN", cuda=False)
        model = PCALayer(2, inter, cfg["alpha"])
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    def predict(nodes):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(nodes), cfg["batch"]):
                b = list(map(int, nodes[s:s + cfg["batch"]]))
                prob, _ = model.to_prob(b, torch.LongTensor(labels[b]), train_flag=False)  # as in test_care/test_pcgnn
                out.append(prob[:, 1].numpy())
        model.train()
        return np.concatenate(out)

    best, best_state = -1.0, None
    val, test = np.asarray(g["idx_val"]), np.asarray(g["idx_test"])
    for epoch in range(cfg["epochs"]):
        if a.kind == "care":
            sampled = undersample(train_pos, train_neg, scale=1)
            inter.batch_num = int(len(sampled) / cfg["batch"]) + 1
        else:
            sampled = pick_step(idx_train, y_train, g["homo"], size=len(train_pos) * 2)
        random.shuffle(sampled)
        for s in range(0, len(sampled), cfg["batch"]):
            b = sampled[s:s + cfg["batch"]]
            if not b:
                continue
            opt.zero_grad()
            loss = model.loss(b, torch.LongTensor(labels[np.array(b)]))
            loss.backward()
            opt.step()
        if epoch % a.eval_every == 0 or epoch == cfg["epochs"] - 1:
            ap_val = average_precision_score(labels[val], predict(val))
            print(f"[{a.kind}] epoch {epoch} val PR-AUC {ap_val:.4f}", flush=True)
            if ap_val > best:
                best = ap_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    np.savez(a.out, p_val=predict(val), p_test=predict(test), best_val=best)


if __name__ == "__main__":
    main()
