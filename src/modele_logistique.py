"""
étape 7a - régression logistique
projet SMOE.

entrée  : data/processed/shots_split.parquet    (étape 5), via préparation.py
sorties : resultats/logistique.csv              (scores)
          resultats/logistique_coefficients.csv (coefficients)
          figures/07_calibration_logistique.png
          figures/08_coefficients_logistique.png
          modeles/logistique.joblib             (modèle final, avec fatigue)

1. réglage de C par validation croisé PAR MATCH sur l'entraînement (critère : log loss).
2. évaluation sur l'entraînement et la validation, comparaison avec la référence b.
3. courbe de calibration.
4. ablation : même modèle sans les variables de fatigue (5 plis + validation).
5. coefficients, en particulier ceux de la fatigue.
le jeu test n'est pas utilisé ici.
"""

from pathlib import Path
 
import joblib
import matplotlib
matplotlib.use("Agg")  # aucune fenêtre : les figures sont seulement sauvegardées
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
 
from preparation import (CIBLE, FATIGUE, charger_donnees, construire_preprocesseur,
                         evaluer, separer, variables)
 
RES_DIR, FIG_DIR, MOD_DIR = Path("resultats"), Path("figures"), Path("modeles")
VALEURS_C = [0.001, 0.01, 0.1, 1.0, 10.0]
N_PLIS = 5
BLEU, ORANGE, GRIS = "#257de0", "#f47230", "#666666"
 
 
# ---------------------------------------------------------------- modèle
def construire_modele(avec_fatigue: bool = True, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", construire_preprocesseur("logistique", avec_fatigue)),
        ("modele", LogisticRegression(C=C, max_iter=2000)),
    ])
 
 
def colonnes(avec_fatigue: bool) -> list[str]:
    num, cat = variables("logistique", avec_fatigue)
    return num + cat
 
 
# ---------------------------------------------------------------- 1. réglage
def regler_C(train: pd.DataFrame) -> float:
    recherche = GridSearchCV(
        construire_modele(True),
        param_grid={"modele__C": VALEURS_C},
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=N_PLIS),
        n_jobs=-1,
    )
    recherche.fit(train[colonnes(True)], train[CIBLE], groups=train["GAME_ID"])
 
    res = pd.DataFrame(recherche.cv_results_)
    print("  C        log loss moyenne (5 plis)   écart-type")
    for _, r in res.iterrows():
        print(f"  {r['param_modele__C']:<8} {-r['mean_test_score']:.5f}"
              f"                    {r['std_test_score']:.5f}")
    meilleur = recherche.best_params_["modele__C"]
    print(f"  → C retenu : {meilleur}")
    return meilleur
 
 
# ---------------------------------------------------------------- 3. calibration
def figure_calibration(y, p, p_ref, chemin: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color=GRIS, lw=1, label="Calibration parfaite")
    for proba, nom, couleur in ((p_ref, "Référence B (zone)", ORANGE),
                                (p, "Régression logistique", BLEU)):
        frac, moy = calibration_curve(y, proba, n_bins=10, strategy="quantile")
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
 
 
# ---------------------------------------------------------------- 4. ablation
def ablation_par_plis(train: pd.DataFrame, C: float) -> pd.DataFrame:
    """Log loss avec et sans fatigue sur les mêmes 5 plis (par match) de l'entraînement."""
    lignes = []
    plis = GroupKFold(n_splits=N_PLIS).split(train, groups=train["GAME_ID"])
    for i, (idx_a, idx_b) in enumerate(plis):
        a, b = train.iloc[idx_a], train.iloc[idx_b]
        scores = {}
        for avec in (True, False):
            m = construire_modele(avec, C).fit(a[colonnes(avec)], a[CIBLE])
            scores[avec] = log_loss(b[CIBLE], m.predict_proba(b[colonnes(avec)])[:, 1])
        lignes.append({"pli": i, "avec_fatigue": scores[True], "sans_fatigue": scores[False],
                       "gain": scores[False] - scores[True]})
    return pd.DataFrame(lignes)
 
 
# ---------------------------------------------------------------- 5. coefficients
def tableau_coefficients(modele: Pipeline) -> pd.DataFrame:
    noms = modele.named_steps["prep"].get_feature_names_out()
    coefs = modele.named_steps["modele"].coef_[0]
    t = pd.DataFrame({"variable": noms, "coefficient": coefs})
    t["rapport_cotes"] = np.exp(t["coefficient"])
    t["fatigue"] = t["variable"].apply(lambda v: any(v.startswith(f) for f in FATIGUE))
    return t.reindex(t["coefficient"].abs().sort_values(ascending=False).index)
 
 
def effets_fatigue(modele: Pipeline, train: pd.DataFrame) -> None:
    """Traduit les coefficients numériques de fatigue en unités concrètes."""
    num, _ = variables("logistique", True)
    echelle = modele.named_steps["prep"].named_transformers_["num"].named_steps["echelle"]
    coefs = modele.named_steps["modele"].coef_[0]
    unites = {"MINUTES_JOUEES": (10, "10 minutes jouées de plus"),
              "MATCHS_7J": (1, "1 match de plus dans les 7 jours"),
              "DEBUT_HISTORIQUE": (1, "premier match du joueur")}
    print("  Effet sur les cotes de réussite (toutes autres choses égales) :")
    for var, (pas, texte) in unites.items():
        j = num.index(var)
        coef_unite = coefs[j] / echelle.scale_[j] * pas
        print(f"    {texte:<38} : cotes × {np.exp(coef_unite):.4f} "
              f"({(np.exp(coef_unite) - 1) * 100:+.2f} %)")
 
 
def figure_coefficients(t: pd.DataFrame, chemin: Path) -> None:
    t = t.iloc[::-1]  # plus grands en haut
    couleurs = [ORANGE if f else BLEU for f in t["fatigue"]]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.5))
    ax.barh(t["variable"], t["coefficient"], color=couleurs, height=0.65)
    ax.axvline(0, color=GRIS, lw=1)
    ax.set_xlabel("Coefficient (variables numériques mises à l'échelle)")
    ax.set_title("Coefficients de la régression logistique", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLEU),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Contexte du tir", "Fatigue"], frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(chemin, dpi=200)
    plt.close(fig)
    print(f"  Figure sauvegardée : {chemin}")
 
 
# ---------------------------------------------------------------- programme
def main() -> None:
    for d in (RES_DIR, FIG_DIR, MOD_DIR):
        d.mkdir(exist_ok=True)
 
    tirs = charger_donnees()
    train, val, _ = separer(tirs)
    print(f"Entraînement : {len(train):,} tirs | validation : {len(val):,} tirs")
 
    print("\n1. Réglage de C (validation croisée par match, 5 plis)")
    C = regler_C(train)
 
    print("\n2. Évaluation")
    modele = construire_modele(True, C).fit(train[colonnes(True)], train[CIBLE])
    modele_sf = construire_modele(False, C).fit(train[colonnes(False)], train[CIBLE])
    p_train = modele.predict_proba(train[colonnes(True)])[:, 1]
    p_val = modele.predict_proba(val[colonnes(True)])[:, 1]
    p_val_sf = modele_sf.predict_proba(val[colonnes(False)])[:, 1]
 
    ref = pd.read_csv(RES_DIR / "references.csv")
    resultats = pd.concat([ref, pd.DataFrame([
        evaluer(train[CIBLE], p_train, train["VALEUR_TIR"], "Logistique — ENTRAÎNEMENT"),
        evaluer(val[CIBLE], p_val, val["VALEUR_TIR"], "Logistique (avec fatigue)"),
        evaluer(val[CIBLE], p_val_sf, val["VALEUR_TIR"], "Logistique (sans fatigue)"),
    ])], ignore_index=True)
    print(resultats.round(4).to_string(index=False))
    ll = resultats.set_index("modele")["log_loss"]
    ecart = ll["Logistique — ENTRAÎNEMENT"] - ll["Logistique (avec fatigue)"]
    print(f"\n  Écart entraînement − validation (log loss) : {ecart:+.4f} "
          f"(proche de 0 = pas de surapprentissage)")
 
    print("\n3. Calibration")
    p_ref = val.join(train.groupby(["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"])
                     [CIBLE].mean().rename("REF"),
                     on=["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"])["REF"]
    p_ref = p_ref.fillna(train[CIBLE].mean())
    figure_calibration(val[CIBLE], p_val, p_ref, FIG_DIR / "07_calibration_logistique.png")
 
    print("\n4. Ablation de la fatigue (5 plis par match de l'entraînement)")
    abl = ablation_par_plis(train, C)
    print(abl.round(5).to_string(index=False))
    print(f"  Gain moyen de la fatigue : {abl['gain'].mean():+.5f} "
          f"(écart-type {abl['gain'].std():.5f}) ; "
          f"la fatigue améliore la log loss sur {(abl['gain'] > 0).sum()} pli(s) sur {N_PLIS}")
    print(f"  Sur la validation : gain de {ll['Logistique (sans fatigue)'] - ll['Logistique (avec fatigue)']:+.5f}")
 
    print("\n5. Coefficients")
    coefs = tableau_coefficients(modele)
    print(coefs.round(4).to_string(index=False))
    effets_fatigue(modele, train)
    figure_coefficients(coefs, FIG_DIR / "08_coefficients_logistique.png")
 
    resultats.to_csv(RES_DIR / "logistique.csv", index=False)
    abl.to_csv(RES_DIR / "logistique_ablation.csv", index=False)
    coefs.to_csv(RES_DIR / "logistique_coefficients.csv", index=False)
    joblib.dump(modele, MOD_DIR / "logistique.joblib")
    print(f"\nSauvegardé : {RES_DIR / 'logistique.csv'}, {RES_DIR / 'logistique_ablation.csv'}, "
          f"{RES_DIR / 'logistique_coefficients.csv'}, {MOD_DIR / 'logistique.joblib'}")
 
 
if __name__ == "__main__":
    main()