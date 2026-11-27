"""
étape 7b - fôret aléatoire
projet SMOE.

entrée  : data/processed/shots_split.parquet    (étape 5), via preparation.py
sorties : resultats/foret.csv, resultats/foret_ablation.csv, resultats/foret_importance.csv
          figures/09_importance_foret.png
          figures/10_dependance_fatigue_foret.png
          figures/11_calibration_foret.png
          modeles/foret.joblib                  (forêt + calibration isotonique)

utilisation :
    python src/modele_foret.py          # version complète (~ 10 à 30 min selon l'ordinateur)
    python sr/modele_foret.py --rapide  # moins d'arbres et de combinaisons, pour tester

1. réglage des hyperparamètres (recherche aléatoire, validation croisée PAR MATCH).
2. prédictions hors pli sur l'entraînement (5 plis par match) -> calibration isotonique
   et ablation de la fatigue sur les même plis.
3. modèle final, évaluation sur la validation (brut et calibré).
4. importance des variables (permutation + impurité).
5. dépendance partielle des variables de fatigue (effets non linéaires).
6. courbe de calibration.
le jeu test n'est pas utilisé ici.
"""

import argparse
import time
from pathlib import Path
 
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
 
from preparation import (CIBLE, FATIGUE, charger_donnees, construire_preprocesseur,
                         evaluer, separer, variables)
 
RES_DIR, FIG_DIR, MOD_DIR = Path("resultats"), Path("figures"), Path("modeles")
BLEU, ORANGE, AQUA, GRIS = "#257de0", "#f47230", "#00b894", "#666666"
GRAINE = 42
N_PLIS = 5
 
GRILLE = {
    "modele__max_depth": [10, 15, 20, None],
    "modele__min_samples_leaf": [20, 50, 100, 200],
    "modele__max_features": ["sqrt", 0.3, 0.5],
}
 
 
# ---------------------------------------------------------------- modèle
def colonnes(avec_fatigue: bool) -> list[str]:
    num, cat = variables("arbres", avec_fatigue)
    return num + cat
 
 
def construire_modele(avec_fatigue: bool, n_arbres: int, **params) -> Pipeline:
    foret = RandomForestClassifier(
        n_estimators=n_arbres,
        max_samples=0.5,        # chaque arbre voit 50 % des tirs : plus rapide, moins de surapprentissage
        n_jobs=-1,
        random_state=GRAINE,
    )
    modele = Pipeline([("prep", construire_preprocesseur("arbres", avec_fatigue)),
                       ("modele", foret)])
    return modele.set_params(**params)
 
 
# ---------------------------------------------------------------- 1. réglage
def regler(train: pd.DataFrame, n_arbres: int, n_iter: int) -> dict:
    recherche = RandomizedSearchCV(
        construire_modele(True, n_arbres),
        param_distributions=GRILLE,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=3),
        random_state=GRAINE,
        n_jobs=1,               # la forêt utilise déjà tous les cœurs
        verbose=1,
    )
    recherche.fit(train[colonnes(True)], train[CIBLE], groups=train["GAME_ID"])
 
    res = pd.DataFrame(recherche.cv_results_).sort_values("rank_test_score")
    print("  max_depth  min_samples_leaf  max_features   log loss (3 plis)")
    for _, r in res.iterrows():
        print(f"  {str(r['param_modele__max_depth']):<10} {r['param_modele__min_samples_leaf']:<17} "
              f"{str(r['param_modele__max_features']):<13} {-r['mean_test_score']:.5f}")
    print(f"  → Retenu : {recherche.best_params_}")
    return recherche.best_params_
 
 
# ---------------------------------------------------------------- 2. hors pli
def predictions_hors_pli(train: pd.DataFrame, avec_fatigue: bool, n_arbres: int,
                         params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Chaque tir de l'entraînement est prédit par une forêt qui n'a jamais vu son match."""
    p = np.zeros(len(train))
    pli = np.zeros(len(train), dtype=int)
    X, y = train[colonnes(avec_fatigue)], train[CIBLE]
    base = construire_modele(avec_fatigue, n_arbres, **params)
    for i, (a, b) in enumerate(GroupKFold(n_splits=N_PLIS).split(X, groups=train["GAME_ID"])):
        m = clone(base).fit(X.iloc[a], y.iloc[a])
        p[b] = m.predict_proba(X.iloc[b])[:, 1]
        pli[b] = i
    return p, pli
 
 
def calibrateur(p_hors_pli: np.ndarray, y: pd.Series) -> IsotonicRegression:
    """Régression isotonique : corrige les probabilités sans changer leur ordre."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    return iso.fit(p_hors_pli, y)
 
 
# ---------------------------------------------------------------- 4. importance
def importance(modele: Pipeline, val: pd.DataFrame, n_echantillon: int) -> pd.DataFrame:
    cols = colonnes(True)
    ech = val.sample(min(n_echantillon, len(val)), random_state=GRAINE)
    perm = permutation_importance(modele, ech[cols], ech[CIBLE], scoring="neg_log_loss",
                                  n_repeats=5, random_state=GRAINE, n_jobs=1)
 
    # Importance par impureté (feature_importances_), regroupée par variable d'origine
    noms = modele.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(modele.named_steps["modele"].feature_importances_, index=noms)
    impurete = {c: imp[[n for n in noms if n == c or n.startswith(c + "_")]].sum() for c in cols}
 
    t = pd.DataFrame({"variable": cols,
                      "permutation": perm.importances_mean,
                      "permutation_ecart_type": perm.importances_std,
                      "impurete": [impurete[c] for c in cols]})
    t["fatigue"] = t["variable"].isin(FATIGUE)
    return t.sort_values("permutation", ascending=False).reset_index(drop=True)
 
 
def figure_importance(t: pd.DataFrame, chemin: Path) -> None:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.5))
    ax.barh(t["variable"], t["permutation"], xerr=t["permutation_ecart_type"],
            color=[ORANGE if f else BLEU for f in t["fatigue"]], height=0.65,
            error_kw={"ecolor": GRIS, "lw": 1, "capsize": 2})
    ax.axvline(0, color=GRIS, lw=1)
    ax.set_xlabel("Hausse de la log loss quand la variable est mélangée (validation)")
    ax.set_title("Importance par permutation — forêt aléatoire", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLEU),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Contexte du tir", "Fatigue"], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)
    print(f"  Figure sauvegardée : {chemin}")
 
 
# ---------------------------------------------------------------- 5. dépendance partielle
def figure_dependance(modele: Pipeline, train: pd.DataFrame, chemin: Path,
                      n_echantillon: int) -> None:
    """Probabilité moyenne prédite quand on fixe une variable de fatigue à chaque valeur."""
    cols = colonnes(True)
    ech = train.sample(min(n_echantillon, len(train)), random_state=GRAINE)[cols].copy()
    num, _ = variables("arbres", True)
    ech[num] = ech[num].astype(float)   # partial_dependence refuse les colonnes entières
    a_tracer = [("MINUTES_JOUEES", "Minutes jouées avant le tir"),
                ("JOURS_REPOS", "Jours de repos"),
                ("MATCHS_7J", "Matchs dans les 7 jours précédents")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (var, titre) in zip(axes, a_tracer):
        pd_res = partial_dependence(modele, ech, [var], kind="average", grid_resolution=25,
                                    percentiles=(0.01, 0.99))
        x, y = pd_res["grid_values"][0], pd_res["average"][0]
        ax.plot(x, y, color=ORANGE, lw=2, marker="o", ms=4)
        print(f"  {var:<15} : de {y.min():.4f} à {y.max():.4f} "
              f"(écart de {(y.max() - y.min()) * 100:.2f} points de %)")
        ax.set_title(titre, loc="left", fontsize=10, fontweight="bold")
        ax.grid(color="#e5e5e5")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)
    axes[0].set_ylabel("Probabilité moyenne prédite")
    fig.suptitle("Dépendance partielle des variables de fatigue — forêt aléatoire",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)
    print(f"  Figure sauvegardée : {chemin}")
 
 
# ---------------------------------------------------------------- 6. calibration
def figure_calibration(y, courbes: list[tuple[np.ndarray, str, str]], chemin: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color=GRIS, lw=1, label="Calibration parfaite")
    for p, nom, couleur in courbes:
        frac, moy = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(moy, frac, marker="o", lw=2, color=couleur, label=nom)
    ax.set_xlabel("Probabilité prédite (moyenne du groupe)")
    ax.set_ylabel("Fréquence réelle de réussite")
    ax.set_title("Courbe de calibration — validation", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#e5e5e5")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)
    print(f"  Figure sauvegardée : {chemin}")
 
 
# ---------------------------------------------------------------- programme
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapide", action="store_true", help="version courte pour tester")
    args = parser.parse_args()
    n_arbres_recherche, n_arbres, n_iter, n_ech = (
        (50, 100, 3, 3000) if args.rapide else (150, 300, 8, 10000))
 
    for d in (RES_DIR, FIG_DIR, MOD_DIR):
        d.mkdir(exist_ok=True)
    debut = time.time()
 
    tirs = charger_donnees()
    train, val, _ = separer(tirs)
    print(f"Entraînement : {len(train):,} tirs | validation : {len(val):,} tirs"
          f"{'  [mode rapide]' if args.rapide else ''}")
 
    print(f"\n1. Réglage ({n_iter} combinaisons × 3 plis par match, {n_arbres_recherche} arbres)")
    params = regler(train, n_arbres_recherche, n_iter)
 
    print(f"\n2. Prédictions hors pli ({N_PLIS} plis par match, {n_arbres} arbres) — avec et sans fatigue")
    p_oof, pli = predictions_hors_pli(train, True, n_arbres, params)
    p_oof_sf, _ = predictions_hors_pli(train, False, n_arbres, params)
    iso = calibrateur(p_oof, train[CIBLE])
    iso_sf = calibrateur(p_oof_sf, train[CIBLE])
 
    abl = pd.DataFrame([{
        "pli": i,
        "avec_fatigue": log_loss(train[CIBLE][pli == i], p_oof[pli == i]),
        "sans_fatigue": log_loss(train[CIBLE][pli == i], p_oof_sf[pli == i]),
    } for i in range(N_PLIS)])
    abl["gain"] = abl["sans_fatigue"] - abl["avec_fatigue"]
    print("  Ablation de la fatigue (log loss hors pli, par pli) :")
    print(abl.round(5).to_string(index=False))
    print(f"  Gain moyen : {abl['gain'].mean():+.5f} (écart-type {abl['gain'].std():.5f}) ; "
          f"gain positif sur {(abl['gain'] > 0).sum()} pli(s) sur {N_PLIS}")
 
    print(f"\n3. Modèle final ({n_arbres} arbres) et évaluation")
    modele = construire_modele(True, n_arbres, **params).fit(train[colonnes(True)], train[CIBLE])
    modele_sf = construire_modele(False, n_arbres, **params).fit(train[colonnes(False)], train[CIBLE])
    p_train = modele.predict_proba(train[colonnes(True)])[:, 1]
    p_val = modele.predict_proba(val[colonnes(True)])[:, 1]
    p_val_cal = iso.predict(p_val)
    p_val_sf_cal = iso_sf.predict(modele_sf.predict_proba(val[colonnes(False)])[:, 1])
 
    precedents = pd.read_csv(RES_DIR / "logistique.csv")
    precedents = precedents[~precedents["modele"].str.contains("ENTRAÎNEMENT")]
    resultats = pd.concat([precedents, pd.DataFrame([
        evaluer(train[CIBLE], p_train, train["VALEUR_TIR"], "Forêt — ENTRAÎNEMENT (vu)"),
        evaluer(train[CIBLE], p_oof, train["VALEUR_TIR"], "Forêt — ENTRAÎNEMENT (hors pli)"),
        evaluer(val[CIBLE], p_val, val["VALEUR_TIR"], "Forêt brute (avec fatigue)"),
        evaluer(val[CIBLE], p_val_cal, val["VALEUR_TIR"], "Forêt calibrée (avec fatigue)"),
        evaluer(val[CIBLE], p_val_sf_cal, val["VALEUR_TIR"], "Forêt calibrée (sans fatigue)"),
    ])], ignore_index=True)
    print(resultats.round(4).to_string(index=False))
    ll = resultats.set_index("modele")["log_loss"]
    print(f"\n  Gain de la fatigue sur la validation (calibrée) : "
          f"{ll['Forêt calibrée (sans fatigue)'] - ll['Forêt calibrée (avec fatigue)']:+.5f}")
 
    print("\n4. Importance des variables (permutation sur un échantillon de la validation)")
    imp = importance(modele, val, n_ech)
    print(imp.round(5).to_string(index=False))
    figure_importance(imp, FIG_DIR / "09_importance_foret.png")
 
    print("\n5. Dépendance partielle des variables de fatigue")
    figure_dependance(modele, train, FIG_DIR / "10_dependance_fatigue_foret.png", n_ech // 2)
 
    print("\n6. Calibration")
    courbes = [(p_val, "Forêt brute", ORANGE), (p_val_cal, "Forêt calibrée", BLEU)]
    if (MOD_DIR / "logistique.joblib").exists():
        logit = joblib.load(MOD_DIR / "logistique.joblib")
        num, cat = variables("logistique", True)
        courbes.append((logit.predict_proba(val[num + cat])[:, 1], "Régression logistique", AQUA))
    figure_calibration(val[CIBLE], courbes, FIG_DIR / "11_calibration_foret.png")
 
    resultats.to_csv(RES_DIR / "foret.csv", index=False)
    abl.to_csv(RES_DIR / "foret_ablation.csv", index=False)
    imp.to_csv(RES_DIR / "foret_importance.csv", index=False)
    joblib.dump({"foret": modele, "calibration": iso, "params": params},
                MOD_DIR / "foret.joblib", compress=3)
    print(f"\nSauvegardé : {RES_DIR / 'foret.csv'}, {RES_DIR / 'foret_ablation.csv'}, "
          f"{RES_DIR / 'foret_importance.csv'}, {MOD_DIR / 'foret.joblib'}")
    print(f"Durée totale : {(time.time() - debut) / 60:.1f} min")
 
 
if __name__ == "__main__":
    main()