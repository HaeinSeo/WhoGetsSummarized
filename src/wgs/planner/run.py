"""Planner entry points: frequency baselines, AEP training (gold-gated), AEP inference."""
from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import pandas as pd

from .. import data
from ..config import Config
from ..storage import log_step, read_jsonl
from .features import candidate_table, embed_features
from .select import frequency_scores, write_plans

log = logging.getLogger(__name__)


def _features(cfg: Config, judge: str, scope: str):
    from ..retrieval import Embedder
    emb = Embedder(cfg)
    table = candidate_table(cfg, judge, scope).reset_index(drop=True)
    X = embed_features(cfg, table, emb)
    d = emb.model.config.hidden_size
    cand_emb = X[:, :d]
    return table, X, cand_emb, emb


def plan_frequency(cfg: Config, judge: str, scope: str, verified: bool = True) -> pd.DataFrame:
    table, _, cand_emb, emb = _features(cfg, judge, scope)
    emb.save()
    scores, eligible = frequency_scores(table, verified)
    name = "frequency" if verified else "frequency_unverified"
    return write_plans(cfg, name, table, scores, cand_emb, eligible)


def _training_products(cfg, table, X, judge, scope, emb):
    import torch
    lab_path = cfg.annotations_dir / "selection_gold.parquet"
    if not lab_path.exists():
        raise FileNotFoundError("AEP training needs human selection labels (selection_gold.parquet). "
                                "Refusing to train or report AEP performance without gold.")
    lab = pd.read_parquet(lab_path).set_index("cand_id")
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    idx_of = {h: i for i, h in enumerate(table.hyp_id)}
    prods = {}
    for bid, idx in table.groupby("bundle_id").indices.items():
        idx = np.asarray(idx)
        hyp = table.hyp_id.iloc[idx]
        m = hyp.isin(lab.index).to_numpy()
        v = np.array([float(lab.at[h, "v_evidence_valid"]) if ok else 0 for h, ok in zip(hyp, m)])
        y = np.array([float(lab.at[h, "y_include"]) if ok else 0 for h, ok in zip(hyp, m)])
        prods[bid] = {"bundle_id": bid, "split": bundles[bid]["split"], "idx": idx,
                      "x": torch.tensor(X[idx]), "v": torch.tensor(v, dtype=torch.float32),
                      "y": torch.tensor(y, dtype=torch.float32), "label_mask": torch.tensor(m),
                      "inv": [], "remove": []}
    # controlled pairs (human-validated only)
    ivg = cfg.annotations_dir / "intervention_gold.parquet"
    pairs = pd.DataFrame(read_jsonl(cfg.data_dir / "intervention_pairs.jsonl"))
    if ivg.exists() and len(pairs):
        ok = set(pd.read_parquet(ivg).query("valid_pair == '1'").item_id)
        pairs = pairs[pairs.pair_id.isin(ok)]
        d = emb.model.config.hidden_size
        probes = candidate_table(cfg, judge, scope, kind="probe").set_index("hyp_id")
        for p in pairs.itertuples():
            src, mod = prods.get(p.original_bundle), prods.get(p.modified_bundle)
            if src is None or p.target_cand_id not in idx_of:
                continue
            k = int(np.where(src["idx"] == idx_of[p.target_cand_id])[0][0])
            if p.type == "target_preserved" and mod is not None:
                sims = X[mod["idx"], :d] @ X[idx_of[p.target_cand_id], :d]
                if sims.max() >= float(cfg["retrieval"]["candidate_merge_threshold"]):
                    src["inv"].append((mod["x"], k, int(sims.argmax())))
            elif p.type == "target_removed" and p.pair_id in probes.index:
                pr = embed_features(cfg, probes.loc[[p.pair_id]].reset_index().assign(
                    aspect=table.aspect.iloc[idx_of[p.target_cand_id]]), emb)
                base = mod["x"] if mod is not None else torch.zeros(0, X.shape[1])
                src["remove"].append((torch.cat([base, torch.tensor(pr)]), base.shape[0]))
    return prods


def train_aep(cfg: Config, judge: str, scope: str, variant: str = "full",
              lam_inv: float | None = None, lam_remove: float | None = None) -> pd.DataFrame:
    import torch
    from .aep import AEP, losses, n_params
    started = dt.datetime.now()
    pc = cfg["planner"]
    lam_inv = pc["lambda_inv"] if lam_inv is None else lam_inv
    lam_remove = pc["lambda_remove"] if lam_remove is None else lam_remove
    table, X, _, emb = _features(cfg, judge, scope)
    prods = _training_products(cfg, table, X, judge, scope, emb)
    emb.save()
    train = [p for p in prods.values() if p["split"] == "train" and p["label_mask"].any()]
    dev = [p for p in prods.values() if p["split"] == "dev" and p["label_mask"].any()]
    if not train or not dev:
        raise ValueError(f"need labeled train and dev products (train={len(train)}, dev={len(dev)})")
    ckdir = cfg.sub("checkpoints")
    logs = []
    for seed in pc["train_seeds"]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = AEP(X.shape[1], pc["hidden_dim"], pc["attention_heads"], pc["attention_layers"],
                    pc["ffn_dim"], pc["dropout"], variant)
        opt = torch.optim.AdamW(model.parameters(), lr=pc["learning_rate"], weight_decay=pc["weight_decay"])
        best, bad = float("inf"), 0
        tag = f"aep_{variant}_inv{lam_inv}_rm{lam_remove}_s{seed}"
        for epoch in range(pc["max_epochs"]):
            model.train()
            order = rng.permutation(len(train))
            for i in range(0, len(order), pc["batch_products"]):
                opt.zero_grad()
                loss = torch.stack([losses(model, train[j], lam_inv, lam_remove)["total"]
                                    for j in order[i:i + pc["batch_products"]]]).mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                res = [losses(model, p, lam_inv, lam_remove) for p in dev]
                dl = {k: float(torch.stack([torch.as_tensor(x[k]) for x in res]).mean())
                      for k in ("sup", "total")}
            logs.append({"tag": tag, "seed": seed, "epoch": epoch, "dev_sup": dl["sup"], "dev_total": dl["total"]})
            if dl["sup"] < best - 1e-4:
                best, bad = dl["sup"], 0
                torch.save({"state": model.state_dict(), "d_in": X.shape[1], "variant": variant,
                            "planner_cfg": pc, "lam_inv": lam_inv, "lam_remove": lam_remove,
                            "seed": seed, "n_params": n_params(model)}, ckdir / f"{tag}.pt")
            else:
                bad += 1
                if bad >= pc.get("patience", 3):
                    break
    df = pd.DataFrame(logs)
    df.to_csv(cfg.out_dir / f"aep_train_log_{variant}.csv", index=False)
    log_step(cfg, f"train-aep-{variant}", {"judge": judge, "scope": scope, "lam_inv": lam_inv,
                                           "lam_remove": lam_remove},
             {"train_products": len(train), "dev_products": len(dev)}, started)
    return df


def plan_aep(cfg: Config, checkpoint: str, judge: str, scope: str, name: str | None = None) -> pd.DataFrame:
    import torch
    from .aep import AEP
    ck = torch.load(checkpoint, map_location="cpu")
    pc = ck["planner_cfg"]
    model = AEP(ck["d_in"], pc["hidden_dim"], pc["attention_heads"], pc["attention_layers"],
                pc["ffn_dim"], pc["dropout"], ck["variant"])
    model.load_state_dict(ck["state"])
    model.eval()
    table, X, cand_emb, emb = _features(cfg, judge, scope)
    emb.save()
    r_all, q_all, s_all = np.zeros(len(table)), np.zeros(len(table)), np.zeros(len(table))
    with torch.no_grad():
        for _, idx in table.groupby("bundle_id").indices.items():
            r, q, s = model(torch.tensor(X[idx]))
            r_all[idx], q_all[idx], s_all[idx] = r.numpy(), q.numpy(), s.numpy()
    eligible = r_all >= 0.5
    return write_plans(cfg, name or f"aep_{ck['variant']}", table, s_all, cand_emb, eligible, r_all, q_all)
