"""
étape 3 - création des variables
projet SMOE

entrées  : data/processed/shots_clean.parquet           (étape 2)
           data/raw/playergamelogs_2025_26.parquet      (étape 1b)
           data/raw/rotations_2025_26.parquet           (étape 1c)
sortie   : data/processed/shots_features.parquet

3a. variables du tirs     : TEMPS_RESTANT, TEMPS_ECOULE, ANGLE_TIR, FAMILLE_TIR
3b. repos et charge       : JOURS_REPOS, JOURS_REPOS_CAT, BACK_TO_BACK, MATCHS_7J,
                            DEBUT_HISTORIQUE
3c. fatigue dans le match : MINUTES_JOUEES
"""

from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------- paramètres
SHOTS_FILE = Path("data/processed/shots_clean.parquet")
LOGS_FILE = Path("data/raw/playergamelogs_2025_26.parquet")
ROT_FILE = Path("data/raw/rotations_2025_26.parquet")
OUT_FILE = Path("data/processed/shots_features.parquet")
 
CLE_TIR = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"]
DUREE_QUART = 720          # secondes
DUREE_PROLONGATION = 300   # secondes
DIXIEMES = 10              # les rotations sont en dixièmes de seconde
TOLERANCE_TERRAIN = 10     # 1 s de tolérance (le temps des tirs est arrondi à la seconde)
PLAFOND_REPOS = 10         # jours ; valeur aussi utilisée pour le premier match
 
# familles de tirs : la première règle qui correspond l'emporte (l'ordre compte)
FAMILLES = [
    ("Putback",          ["tip", "putback"]),
    ("Dunk",             ["dunk"]),
    ("Hook",             ["hook"]),
    ("Floater",          ["floating", "floater"]),
    ("Layup",            ["layup", "finger roll"]),
    ("Jumper difficile", ["step back", "step-back", "fadeaway", "turnaround"]),
    ("Pull-up",          ["pullup", "pull-up", "pull up"]),
    ("Jump shot",        ["jump shot", "jumper", "jump bank"]),
]


def charger() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tirs = pd.read_parquet(SHOTS_FILE)
    logs = pd.read_parquet(LOGS_FILE)
    rot = pd.read_parquet(ROT_FILE)
    for df in (tirs, logs, rot):
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.normalize()
    return tirs, logs, rot


# ================================================================ 3a. variables du tir
def temps_ecoule(tirs: pd.DataFrame) -> pd.DataFrame:
    """temps écoulé depuis le début du match, en secondes"""
    p = tirs["PERIOD"]
    tr = tirs["MINUTES_REMAINING"] * 60 + tirs["SECONDS_REMAINING"]
    reglementaire = (p - 1) * DUREE_QUART + (DUREE_QUART - tr)
    prolongation = 4 * DUREE_QUART + (p - 5) * DUREE_PROLONGATION + (DUREE_PROLONGATION - tr)
    tirs["TEMPS_RESTANT"] = tr
    tirs["TEMPS_ECOULE"] = np.where(p <= 4, reglementaire, prolongation)

    print(f"    TEMPS_ECOULE : de {tirs['TEMPS_ECOULE'].min()} à {tirs['TEMPS_ECOULE'].max()} s "
          f";   valeurs négatives : {(tirs['TEMPS_ECOULE'] < 0).sum()} (attendu 0)")
    return tirs

def angle_tir(tirs: pd.DataFrame) -> pd.DataFrame:
    """Angle par rapport à l'axe du panier : 0° = face au panier, ±90° = ligne de fond."""
    tirs["ANGLE_TIR"] = np.degrees(np.arctan2(tirs["LOC_X"], tirs["LOC_Y"])).abs()
    print(f"  ANGLE_TIR : de {tirs['ANGLE_TIR'].min():.1f}° à {tirs['ANGLE_TIR'].max():.1f}°")
    return tirs
 
 
def famille_tir(tirs: pd.DataFrame) -> pd.DataFrame:
    """Regroupe ACTION_TYPE (≈ 50 valeurs) en ~8 familles."""
    texte = tirs["ACTION_TYPE"].str.lower()
    famille = pd.Series("Autre", index=tirs.index)
    non_classe = pd.Series(True, index=tirs.index)
    for nom, mots in FAMILLES:
        correspond = texte.str.contains("|".join(mots), regex=True) & non_classe
        famille[correspond] = nom
        non_classe &= ~correspond
    tirs["FAMILLE_TIR"] = famille
 
    resume = tirs.groupby("FAMILLE_TIR").agg(tirs=("SHOT_MADE_FLAG", "size"),
                                             reussite=("SHOT_MADE_FLAG", "mean"),
                                             types=("ACTION_TYPE", "nunique"))
    print("  Familles de tirs :")
    print(resume.sort_values("tirs", ascending=False).round(3).to_string())
 
    autres = tirs.loc[tirs["FAMILLE_TIR"] == "Autre", "ACTION_TYPE"].value_counts()
    if len(autres):
        print("  ACTION_TYPE classés « Autre » (à vérifier) :")
        print(autres.to_string())
    return tirs
 
 
# ================================================================ 3b. repos et charge
def repos_et_charge(logs: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (joueur, match) avec les variables de repos et de charge."""
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
 
    precedent = logs.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    logs["DEBUT_HISTORIQUE"] = precedent.isna().astype(int)
    jours = (logs["GAME_DATE"] - precedent).dt.days - 1
    logs["JOURS_REPOS"] = jours.fillna(PLAFOND_REPOS).clip(upper=PLAFOND_REPOS).astype(int)
    logs["BACK_TO_BACK"] = (logs["JOURS_REPOS"] == 0).astype(int)
    logs["JOURS_REPOS_CAT"] = pd.cut(logs["JOURS_REPOS"], bins=[-1, 0, 1, 2, np.inf],
                                     labels=["0", "1", "2", "3+"]).astype(str)
 
    # MATCHS_7J : matchs du joueur dans [date - 7 j, date - 1 j]
    def compter_7j(dates: pd.Series) -> np.ndarray:
        d = dates.to_numpy()
        return (np.searchsorted(d, d, side="left")
                - np.searchsorted(d, d - np.timedelta64(7, "D"), side="left"))
 
    logs["MATCHS_7J"] = np.concatenate(
        [compter_7j(g) for _, g in logs.groupby("PLAYER_ID", sort=False)["GAME_DATE"]])
 
    print(f"  Jours de repos négatifs : {(jours < 0).sum()} (attendu 0 : deux matchs le même jour ?)")
    print("  Répartition de JOURS_REPOS_CAT (joueur × match) :")
    print(logs["JOURS_REPOS_CAT"].value_counts().sort_index().to_string())
    print(f"  BACK_TO_BACK : {logs['BACK_TO_BACK'].mean():.1%} des matchs joués")
    print(f"  MATCHS_7J    : de {logs['MATCHS_7J'].min()} à {logs['MATCHS_7J'].max()} "
          f"(attendu de 0 à 4 ou 5)")
    return logs[["GAME_ID", "PLAYER_ID", "GAME_DATE", "JOURS_REPOS", "JOURS_REPOS_CAT",
                 "BACK_TO_BACK", "MATCHS_7J", "DEBUT_HISTORIQUE"]]
 
 
def joindre_repos(tirs: pd.DataFrame, repos: pd.DataFrame) -> pd.DataFrame:
    avant = len(tirs)
    tirs = tirs.merge(repos.drop(columns="GAME_DATE"), on=["GAME_ID", "PLAYER_ID"],
                      how="left", validate="many_to_one")
    manquants = tirs["JOURS_REPOS"].isna().sum()
    print(f"  Jointure : {avant:,} tirs avant, {len(tirs):,} après ; "
          f"sans journal : {manquants} (attendu 0)")
    return tirs
 
 
# ================================================================ 3c. minutes jouées
def minutes_jouees(tirs: pd.DataFrame, rot: pd.DataFrame) -> pd.DataFrame:
    """
    Minutes jouées par le tireur avant le tir =
    somme des séjours terminés avant le tir + portion écoulée du séjour en cours.
    """
    t = tirs[CLE_TIR + ["TEMPS_ECOULE"]].copy()
    t["T_TIR"] = t["TEMPS_ECOULE"] * DIXIEMES                     # en dixièmes de seconde
 
    sejours = rot.rename(columns={"PERSON_ID": "PLAYER_ID"})[
        ["GAME_ID", "PLAYER_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]]
    m = t.merge(sejours, on=["GAME_ID", "PLAYER_ID"], how="left")
 
    # Contribution de chaque séjour : de IN jusqu'à min(OUT, moment du tir), jamais négative
    fin = np.minimum(m["OUT_TIME_REAL"], m["T_TIR"])
    m["CONTRIB"] = (fin - m["IN_TIME_REAL"]).clip(lower=0)
    # Le tireur est-il sur le terrain au moment du tir ?
    m["SUR_TERRAIN"] = ((m["IN_TIME_REAL"] - TOLERANCE_TERRAIN <= m["T_TIR"])
                        & (m["T_TIR"] <= m["OUT_TIME_REAL"] + TOLERANCE_TERRAIN))
 
    agg = m.groupby(CLE_TIR, sort=False).agg(
        CONTRIB=("CONTRIB", "sum"),
        SUR_TERRAIN=("SUR_TERRAIN", "any"),
        N_SEJOURS=("IN_TIME_REAL", "count"),
    ).reset_index()
    agg["MINUTES_JOUEES"] = agg["CONTRIB"] / (DIXIEMES * 60)
 
    tirs = tirs.merge(agg[CLE_TIR + ["MINUTES_JOUEES", "SUR_TERRAIN", "N_SEJOURS"]],
                      on=CLE_TIR, how="left", validate="one_to_one")
 
    sans_sejour = (tirs["N_SEJOURS"] == 0).sum()
    hors_terrain = (~tirs["SUR_TERRAIN"]).sum()
    trop = (tirs["MINUTES_JOUEES"] > tirs["TEMPS_ECOULE"] / 60 + 0.05).sum()
    print(f"  Tirs sans aucun séjour du tireur      : {sans_sejour} (attendu 0)")
    print(f"  Tireur absent du terrain au moment du tir : {hors_terrain} (attendu 0 ou presque)")
    print(f"  MINUTES_JOUEES > temps écoulé         : {trop} (attendu 0)")
    print(f"  MINUTES_JOUEES : min {tirs['MINUTES_JOUEES'].min():.2f}, "
          f"médiane {tirs['MINUTES_JOUEES'].median():.2f}, max {tirs['MINUTES_JOUEES'].max():.2f}")
    return tirs.drop(columns=["N_SEJOURS"])
 
 
# ================================================================ aperçu de la fatigue
def apercu_fatigue(tirs: pd.DataFrame) -> None:
    """Premier regard (brut, non ajusté) sur la fatigue et le taux de réussite."""
    print("  Taux de réussite selon BACK_TO_BACK :")
    print(tirs.groupby("BACK_TO_BACK")["SHOT_MADE_FLAG"].agg(["size", "mean"])
          .round(3).to_string())
    print("  Taux de réussite selon JOURS_REPOS_CAT :")
    print(tirs.groupby("JOURS_REPOS_CAT")["SHOT_MADE_FLAG"].agg(["size", "mean"])
          .round(3).to_string())
    tranches = pd.cut(tirs["MINUTES_JOUEES"], bins=[0, 6, 12, 18, 24, 30, 36, 60],
                      include_lowest=True)
    print("  Taux de réussite selon MINUTES_JOUEES :")
    print(tirs.groupby(tranches, observed=True)["SHOT_MADE_FLAG"].agg(["size", "mean"])
          .round(3).to_string())
 
 
# ================================================================ programme
def main() -> None:
    tirs, logs, rot = charger()
    print(f"{len(tirs):,} tirs chargés")
 
    print("\n3a. Variables du tir")
    tirs = temps_ecoule(tirs)
    tirs = angle_tir(tirs)
    tirs = famille_tir(tirs)
 
    print("\n3b. Repos et charge")
    repos = repos_et_charge(logs)
    tirs = joindre_repos(tirs, repos)
 
    print("\n3c. Minutes jouées")
    tirs = minutes_jouees(tirs, rot)
 
    print("\nAperçu brut de la fatigue (non ajusté pour la distance ou le type de tir)")
    apercu_fatigue(tirs)
 
    hors_terrain = ~tirs["SUR_TERRAIN"]
    if hors_terrain.any():
        print(f"\n{hors_terrain.sum()} tirs dont le tireur n'est pas sur le terrain sont retirés")
        tirs = tirs[~hors_terrain]
    tirs = tirs.drop(columns=["SUR_TERRAIN"]).reset_index(drop=True)
 
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tirs.to_parquet(OUT_FILE, index=False)
    print(f"\nSauvegardé : {OUT_FILE}  ({len(tirs):,} tirs, {tirs.shape[1]} colonnes)")
    print(f"Colonnes : {', '.join(tirs.columns)}")
 
 
if __name__ == "__main__":
    main()