"""
étape 5 - division des données par match
projet SMOE

entrée  : data/processed/shots_features.parquet     (étape 3)
sorties : data/processed/shots_split.parquet        (tirs + colonnes SPLIT et PLI)
          data/processed/division_matchs.csv        (une ligne par match : GAME_ID, SPLIT, PLI)

1. division d'évaluation : 80% entraînement / 10% validation / 10% test, par match.
2. plis pour le SMOE     : 5 plis par match (validation croisée hors échantillon).
tous les tirs d'un même match restent du même côté (pas de fuite de données).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paramètres
IN_FILE = Path("data/processed/shots_features.parquet")
OUT_FILE = Path("data/processed/shots_split.parquet")
OUT_MATCHS = Path("data/processed/division_matchs.csv")

GRAINE = 42                     # rend la division reproductible
PROPORTIONS = (0.80, 0.10, 0.10)
N_PLIS = 5


# ----------------------------------------------------------------- division
def diviser_matchs(game_ids: np.ndarray) -> pd.DataFrame:
    """attribue à chaque match un SPLIT (train/val/test) et un PLI (0 à N_PLIS-1)."""
    rng = np.random.default_rng(GRAINE)
    melanges = rng.permutation(np.sort(game_ids))  # tri d'abord : même résultat partout

    n = len(melanges)
    n_train = round(PROPORTIONS[0] * n)
    n_val = round(PROPORTIONS[1] * n)
    

    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    split[n_train + n_val:] = "test"

    pli = np.arange(n) % N_PLIS                 # attribution en tournant : plis équilibrés

    return pd.DataFrame({"GAME_ID": melanges, "SPLIT": split, "PLI": pli})


# --------------------------------------------------------------- vérifications
def resume(tirs: pd.DataFrame, colonne: str) -> pd.DataFrame:
    g = tirs.groupby(colonne).agg(
        matchs=("GAME_ID", "nunique"),
        tirs=("SHOT_MADE_FLAG", "size"),
        reussite=("SHOT_MADE_FLAG", "mean"),
        part_3pts=("VALEUR_TIR", lambda v: (v == 3).mean()),
        debut=("GAME_DATE", "min"),
        fin=("GAME_DATE", "max"),
    )
    g["part_tirs"] = g["tirs"] / g["tirs"].sum()
    g["debut"] = g["debut"].dt.date
    g["fin"] = g["fin"].dt.date
    return g[["matchs", "tirs", "part_tirs", "reussite", "part_3pts", "debut", "fin"]]
 
 
def verifier(tirs: pd.DataFrame, division: pd.DataFrame) -> None:
    print("\n=== Contrôle anti-fuite ===")
    ensembles = {s: set(division.loc[division["SPLIT"] == s, "GAME_ID"])
                 for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        commun = ensembles[a] & ensembles[b]
        print(f"  Matchs communs {a} / {b} : {len(commun)}   (attendu 0)")
    matchs_par_tir = tirs.groupby("GAME_ID")["SPLIT"].nunique().max()
    print(f"  Nombre maximal de jeux pour un même match : {matchs_par_tir}   (attendu 1)")
    print(f"  Tirs sans SPLIT ou sans PLI : "
          f"{tirs[['SPLIT', 'PLI']].isna().any(axis=1).sum()}   (attendu 0)")
 
    reussite_globale = tirs["SHOT_MADE_FLAG"].mean()
    print(f"\nTaux de réussite global : {reussite_globale:.4f}")
 
    print("\n=== Division d'évaluation (80 / 10 / 10) ===")
    r = resume(tirs, "SPLIT").loc[["train", "val", "test"]]
    print(r.round(4).to_string())
 
    print(f"\n=== Plis pour le SMOE ({N_PLIS} plis) ===")
    print(resume(tirs, "PLI").round(4).to_string())
 
    ecart_max = (r["reussite"] - reussite_globale).abs().max()
    print(f"\nÉcart maximal du taux de réussite entre un jeu et la moyenne : "
          f"{ecart_max * 100:.2f} point(s) de pourcentage   (attendu < 1)")
 
 
# ---------------------------------------------------------------- programme
def main() -> None:
    tirs = pd.read_parquet(IN_FILE)
    tirs["GAME_ID"] = tirs["GAME_ID"].astype(str).str.zfill(10)
    print(f"{len(tirs):,} tirs, {tirs['GAME_ID'].nunique()} matchs chargés")
 
    division = diviser_matchs(tirs["GAME_ID"].unique())
    tirs = tirs.drop(columns=["SPLIT", "PLI"], errors="ignore")
    tirs = tirs.merge(division, on="GAME_ID", how="left", validate="many_to_one")
 
    verifier(tirs, division)
 
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tirs.to_parquet(OUT_FILE, index=False)
    division.sort_values("GAME_ID").to_csv(OUT_MATCHS, index=False)
    print(f"\nSauvegardé : {OUT_FILE}  ({len(tirs):,} tirs, {tirs.shape[1]} colonnes)")
    print(f"Sauvegardé : {OUT_MATCHS}  ({len(division)} matchs)")
 
 
if __name__ == "__main__":
    main()