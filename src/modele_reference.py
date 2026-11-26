"""
étape 6 - modèles de référence et vérification des préprocesseurs
project SMOE

entrée  : data/processed/shots_split.parquet    (étape 5)
sortie  : resultats/references.csv              (scores des références sur la validation)

référence a : moyenne de la ligue (une seule probabilité pour tous les tirs).
référence b : modèle 0 du rapport - taux de réussite moyen par zone
              (SHOT_ZONE_BASIC x SHOT_ZONE_AREA x SHOT_ZONE_RANGE).
les deux sont calculées sur l'ENTRAÎNEMENT seulement et évaluées sur la VALIDATION.
le jeu test n'est pas utilisé ici.
"""

from pathlib import Path

import pandas as pd

from preparation import (CIBLE, charger_donnees, construire_preprocesseur, evaluer,
                         separer, variables)

OUT_DIR = Path("resultats")
COLONNES_ZONE = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]
MIN_TIRS_ZONE = 30  # zone trop rare à l'entraînement -> moyenne de la ligue


# ---------------------------------------------------------------- références
def reference_moyenne(train: pd.DataFrame, autre: pd.DataFrame) -> pd.Series:
    return pd.Series(train[CIBLE].mean(), index=autre.index)
 
 
def reference_zone(train: pd.DataFrame, autre: pd.DataFrame) -> pd.Series:
    """ref_fg_zone : taux par zone calculé sur l'entraînement, appliqué à un autre jeu."""
    stats = train.groupby(COLONNES_ZONE)[CIBLE].agg(["mean", "size"])
    stats = stats[stats["size"] >= MIN_TIRS_ZONE]["mean"].rename("REF_FG_ZONE")
    print(f"  Zones utilisées : {len(stats)} (au moins {MIN_TIRS_ZONE} tirs à l'entraînement)")
 
    p = autre.join(stats, on=COLONNES_ZONE)["REF_FG_ZONE"]
    manquants = p.isna().sum()
    print(f"  Tirs de validation dont la zone est absente ou rare : {manquants} "
          f"→ moyenne de la ligue")
    return p.fillna(train[CIBLE].mean())
 
 
# ---------------------------------------------------------------- préprocesseurs
def verifier_preprocesseurs(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Ajuste chaque préprocesseur sur l'entraînement et transforme la validation."""
    for modele in ("arbres", "logistique"):
        for avec_fatigue in (True, False):
            num, cat = variables(modele, avec_fatigue)
            prep = construire_preprocesseur(modele, avec_fatigue)
            X_train = prep.fit_transform(train[num + cat])
            X_val = prep.transform(val[num + cat])
            etiquette = f"{modele:<10} {'avec' if avec_fatigue else 'sans'} fatigue"
            print(f"  {etiquette} : {len(num)} num. + {len(cat)} cat. → "
                  f"{X_train.shape[1]} colonnes (train {X_train.shape[0]:,}, val {X_val.shape[0]:,})")
            if modele == "logistique" and avec_fatigue:
                moy = X_train[:, :len(num) + 1].mean(axis=0).round(3)
                print(f"    moyennes des numériques après mise à l'échelle (attendu ≈ 0) : {moy}")
                noms = prep.get_feature_names_out()
                print(f"    colonnes : {', '.join(noms)}")
 
 
# ---------------------------------------------------------------- programme
def main() -> None:
    tirs = charger_donnees()
    train, val, _ = separer(tirs)
    print(f"Entraînement : {len(train):,} tirs | validation : {len(val):,} tirs "
          f"(test non utilisé)")
 
    print("\nRéférence A — moyenne de la ligue")
    p_moy = reference_moyenne(train, val)
    print(f"  Probabilité unique : {p_moy.iloc[0]:.4f}")
 
    print("\nRéférence B — modèle 0 : taux moyen par zone")
    p_zone = reference_zone(train, val)
 
    resultats = pd.DataFrame([
        evaluer(val[CIBLE], p_moy, val["VALEUR_TIR"], "A. Moyenne de la ligue"),
        evaluer(val[CIBLE], p_zone, val["VALEUR_TIR"], "B. Modèle 0 : taux par zone"),
    ])
    print("\n=== Scores sur la validation (les modèles devront battre la ligne B) ===")
    print(resultats.round(4).to_string(index=False))
 
    print("\nVérification des préprocesseurs (ajustés sur l'entraînement seulement)")
    verifier_preprocesseurs(train, val)
 
    OUT_DIR.mkdir(exist_ok=True)
    resultats.to_csv(OUT_DIR / "references.csv", index=False)
    print(f"\nSauvegardé : {OUT_DIR / 'references.csv'}")
 
 
if __name__ == "__main__":
    main()