"""
outils communs du projet SMOE (étapes 6 et suivantes).

à importer depuis les autres scripts de src/ :
    "from preparation import charger_donnees, construire_preprocesseur, evaluer, ..."

contient :
- le chargement des données divisée (étape 5) ;
- les listes de variables (avec ou sans fatigue) ;
- les préprocesseurs scikit-learn (encodage one-hot, mise à l'échelle) ;
- la fonction d'évaluation commune à tous les modèles.
"""

from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
 
DATA_FILE = Path("data/processed/shots_split.parquet")
CIBLE = "SHOT_MADE_FLAG"
 
# ---------------------------------------------------------------- variables
# Variables de fatigue : retirées pour la version « sans fatigue » (ablation)
FATIGUE = ["MINUTES_JOUEES", "JOURS_REPOS", "JOURS_REPOS_CAT", "BACK_TO_BACK",
           "MATCHS_7J", "DEBUT_HISTORIQUE"]
 
# Modèles à base d'arbres : toutes les variables (la multicolinéarité ne les gêne pas)
ARBRES_NUM = ["DIST_CALC", "ANGLE_TIR", "LOC_X", "LOC_Y", "VALEUR_TIR",
              "PERIOD", "TEMPS_RESTANT", "TEMPS_ECOULE",
              "MINUTES_JOUEES", "JOURS_REPOS", "BACK_TO_BACK", "MATCHS_7J", "DEBUT_HISTORIQUE"]
ARBRES_CAT = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "FAMILLE_TIR"]
 
# Régression logistique : jeu réduit validé par le VIF (étape 4)
LOGIT_NUM = ["DIST_CALC", "ANGLE_TIR", "VALEUR_TIR", "TEMPS_RESTANT",
             "MINUTES_JOUEES", "MATCHS_7J", "DEBUT_HISTORIQUE"]
LOGIT_CAT = ["SHOT_ZONE_BASIC", "FAMILLE_TIR", "JOURS_REPOS_CAT"]
 
# Colonnes à garder en plus des variables : regroupement, calcul du SMOE
COLONNES_UTILES = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "GAME_DATE",
                   "SPLIT", "PLI", "VALEUR_TIR", CIBLE]
 
 
def variables(modele: str, avec_fatigue: bool = True) -> tuple[list[str], list[str]]:
    """Renvoie (numériques, catégorielles) pour 'arbres' ou 'logistique'."""
    if modele == "arbres":
        num, cat = ARBRES_NUM, ARBRES_CAT
    elif modele == "logistique":
        num, cat = LOGIT_NUM, LOGIT_CAT
    else:
        raise ValueError("modele doit être 'arbres' ou 'logistique'")
    if not avec_fatigue:
        num = [c for c in num if c not in FATIGUE]
        cat = [c for c in cat if c not in FATIGUE]
    return list(num), list(cat)
 
 
# ---------------------------------------------------------------- données
def charger_donnees() -> pd.DataFrame:
    tirs = pd.read_parquet(DATA_FILE)
    tirs["GAME_ID"] = tirs["GAME_ID"].astype(str).str.zfill(10)
    tirs[CIBLE] = tirs[CIBLE].astype(int)
    tirs["POINTS"] = tirs[CIBLE] * tirs["VALEUR_TIR"]
    return tirs
 
 
def separer(tirs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Renvoie (train, val, test). Le test ne doit servir qu'une fois, à la toute fin."""
    return (tirs[tirs["SPLIT"] == "train"].copy(),
            tirs[tirs["SPLIT"] == "val"].copy(),
            tirs[tirs["SPLIT"] == "test"].copy())
 
 
# ---------------------------------------------------------------- préprocesseurs
def _ajouter_carre(X):
    """Ajoute le carré de la première colonne (la distance) pour la régression logistique."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, X[:, [0]] ** 2])
 
 
def _noms_avec_carre(_, noms):
    return list(noms) + ["DIST_CALC_CARRE"]
 
 
def construire_preprocesseur(modele: str, avec_fatigue: bool = True) -> ColumnTransformer:
    """
    Préprocesseur scikit-learn, à placer dans un Pipeline devant le modèle.
    Il est ajusté (fit) sur l'entraînement seulement, ce qui évite les fuites.
    - arbres     : one-hot des catégories, numériques telles quelles (pas de mise à l'échelle)
    - logistique : distance² ajoutée, mise à l'échelle des numériques, one-hot des catégories
    """
    num, cat = variables(modele, avec_fatigue)
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
 
    if modele == "arbres":
        transformateurs = [("num", "passthrough", num), ("cat", one_hot, cat)]
    else:
        # DIST_CALC doit être la première colonne numérique pour _ajouter_carre
        assert num[0] == "DIST_CALC"
        num_pipe = Pipeline([
            ("carre", FunctionTransformer(_ajouter_carre, feature_names_out=_noms_avec_carre)),
            ("echelle", StandardScaler()),
        ])
        transformateurs = [("num", num_pipe, num), ("cat", one_hot, cat)]
 
    return ColumnTransformer(transformateurs, verbose_feature_names_out=False)
 
 
# ---------------------------------------------------------------- évaluation
def evaluer(y: np.ndarray, p: np.ndarray, valeur: np.ndarray, nom: str = "") -> dict:
    """
    Métriques du rapport préliminaire.
    y : SHOT_MADE_FLAG (0/1) ; p : probabilité prédite ; valeur : 2 ou 3.
    """
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    valeur = np.asarray(valeur)
    points = y * valeur
    xpts = p * valeur
    auc = roc_auc_score(y, p) if np.unique(p).size > 1 else 0.5
    pred = (p >= 0.5).astype(int)
    return {
        "modele": nom,
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": auc,
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "rmse_xpts": float(np.sqrt(np.mean((points - xpts) ** 2))),
        "xpts_moyen": float(xpts.mean()),
        "points_moyens": float(points.mean()),
    }