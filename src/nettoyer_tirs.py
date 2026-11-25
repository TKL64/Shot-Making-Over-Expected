"""
étape 2 - nettoyage des tirs 2025-26
projet SMOE

entrées     : data/raw/shots_2025_26.parquet
              data/raw/rotations_matchs_invalides.csv
sortie      : data/processed/shots_clean.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------- paramètres
SHOTS_FILE = Path("data/raw/shots_2025_26.parquet")
INVALIDES_FILE = Path("data/raw/rotations_matchs_invalides.csv")
OUT_FILE = Path("data/processed/shots_clean.parquet")

COLONNES = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD",
    "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE",
    "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "GAME_DATE",
    "HTM",  # temporaire : sert au contrôle du biais d'aréna, retirée avant la sauvegarde
]
COLONNES_ENTIERES = ["GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD",
                     "MINUTES_REMAINING", "SECONDS_REMAINING", "SHOT_DISTANCE",
                     "LOC_X", "LOC_Y", "SHOT_MADE_FLAG"]
CLE = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"]

DISTANCE_MAX = 40           # pieds
HEAVE_DISTANCE = 36        # pieds  (définition NBA d'un heave, 2025-26)
HEAVE_SECONDES = 3         # dernières secondes des périodes 1 à 3
# géométrie de la ligne de 3 9coordonées en dixièmes de pied, panier à l'origine
COIN_X = 220                # 22 pi : ligne droite des coins
COIN_Y_MAX = 92.5           # hauteur où la ligne droite rejoint l'arc
RAYON_ARC = 237.5           # 23.75 pi
TOLERANCE = 5               # 0.5 pi de tolérance sur les coordonnées
SEUIL_Z_ARENA = 2.0         # écart-types pour signaler une aréna

journal: list[dict] = []    # journal de nettoyage


def noter(etape: str, avant: int, apres: int) -> None:
    journal.append({"étape": etape, "tirs retirés": avant - apres, "tirs restants": apres})
    print(f"    {etape} : {avant - apres:,} retirés, {apres:,} restants")


# ----------------------------------------------------------------- 1. colonnes
def garder_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """garde les colonnes utiles (retire EVENT_TYPE, noms, GRID_TYPE, etc.)"""
    manquantes = [c for c in COLONNES if c not in df.columns]
    if manquantes:
        raise KeyError(f"colonnes absentes du fichier : {manquantes}")
    retirees = [c for c in df.columns if c not in COLONNES]
    print(f"   {len(retirees)} colonnes retirées : {', '.join(retirees)}")
    return df[COLONNES].copy()


# ----------------------------------------------------------------- 2. types
def convertir_types(df: pd.DataFrame) -> pd.DataFrame:
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"].astype(str), format="%Y%m%d")
    for col in COLONNES_ENTIERES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    manquants = df.isna().sum()
    manquants = manquants[manquants > 0]
    if len(manquants):
        print("  !!! Valeurs manquantes :")
        print(manquants.to_string())
    else:
        print("  Aucune valeur manquante")
 
    df[COLONNES_ENTIERES] = df[COLONNES_ENTIERES].astype(int)
 
    types = df["SHOT_TYPE"].unique()
    print(f"  Valeurs de SHOT_TYPE : {list(types)}")
    df["VALEUR_TIR"] = np.where(df["SHOT_TYPE"].str.startswith("3"), 3, 2)
    return df
 
 
# ---------------------------------------------------------------- 3. doublons
def retirer_doublons(df: pd.DataFrame) -> pd.DataFrame:
    avant = len(df)
    df = df.drop_duplicates(subset=CLE)
    noter("Doublons", avant, len(df))
    return df
 
 
# ---------------------------------------------------------------- 4. matchs invalides
def retirer_matchs_invalides(df: pd.DataFrame) -> pd.DataFrame:
    invalides = pd.read_csv(INVALIDES_FILE, dtype={"GAME_ID": str})
    ids = set(invalides["GAME_ID"].str.zfill(10))
    avant = len(df)
    df = df[~df["GAME_ID"].isin(ids)]
    noter(f"Matchs aux rotations invalides ({len(ids)})", avant, len(df))
    return df
 
 
# ---------------------------------------------------------------- 5. tirs lointains
def retirer_tirs_lointains(df: pd.DataFrame) -> pd.DataFrame:
    trop_loin = df["SHOT_DISTANCE"] > DISTANCE_MAX
    arriere = df["SHOT_ZONE_BASIC"] == "Backcourt"
    print(f"  > {DISTANCE_MAX} pi : {trop_loin.sum():,} ; arrière-terrain : {arriere.sum():,} "
          f"; les deux : {(trop_loin & arriere).sum():,}")
    print(f"  Taux de réussite de ces tirs : "
          f"{df.loc[trop_loin | arriere, 'SHOT_MADE_FLAG'].mean():.1%}")
    avant = len(df)
    df = df[~(trop_loin | arriere)]
    noter(f"Tirs à plus de {DISTANCE_MAX} pi ou de l'arrière-terrain", avant, len(df))
    return df


# ---------------------------------------------------------------- 5b. heaves (règle 2025-26)
def retirer_heaves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Règle NBA 2025-26 : un heave raté (≥ 36 pi, 3 dernières secondes des périodes 1 à 3)
    est compté comme tentative de l'équipe, pas du joueur. Les heaves ratés sont donc
    absents des données, mais les réussis y sont. On retire tous les tirs qui répondent
    à la définition, réussis ou non, pour éviter ce biais de sélection.
    """
    temps_restant = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    heave = (df["PERIOD"] <= 3) & (temps_restant <= HEAVE_SECONDES) \
        & (df["SHOT_DISTANCE"] >= HEAVE_DISTANCE)
    print(f"  Tirs répondant à la définition d'un heave : {heave.sum():,} "
          f"(taux de réussite {df.loc[heave, 'SHOT_MADE_FLAG'].mean():.1%})")
    avant = len(df)
    df = df[~heave]
    noter("Heaves de fin de période (règle 2025-26)", avant, len(df))
 
    loin = df["SHOT_DISTANCE"] >= HEAVE_DISTANCE
    print(f"  Tirs restants à ≥ {HEAVE_DISTANCE} pi : {loin.sum():,} "
          f"(taux de réussite {df.loc[loin, 'SHOT_MADE_FLAG'].mean():.1%})")
    return df


# ---------------------------------------------------------------- 6. géométrie
def controler_geometrie(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute DIST_CALC et documente les incohérences (ne retire rien)."""
    x, y = df["LOC_X"], df["LOC_Y"]
    rayon = np.sqrt(x**2 + y**2)            # en dixièmes de pied
    df["DIST_CALC"] = rayon / 10            # en pieds
 
    # a) distance recalculée vs SHOT_DISTANCE (entier fourni par l'API)
    ecart = (df["DIST_CALC"] - df["SHOT_DISTANCE"]).abs()
    print("  a) Distance recalculée vs SHOT_DISTANCE")
    print(f"     écart < 1 pi : {(ecart < 1).mean():.2%} ; "
          f"écart > 2 pi : {(ecart > 2).sum():,} tirs ; écart max : {ecart.max():.1f} pi")
 
    # b) type de tir vs position (avec tolérance)
    dans_coin = y <= COIN_Y_MAX
    surement_3 = (dans_coin & (x.abs() >= COIN_X + TOLERANCE)) | (rayon >= RAYON_ARC + TOLERANCE)
    peut_etre_3 = (dans_coin & (x.abs() >= COIN_X - TOLERANCE)) | (rayon >= RAYON_ARC - TOLERANCE)
 
    est_3 = df["VALEUR_TIR"] == 3
    trois_trop_proche = est_3 & ~peut_etre_3        # « 3 points » à l'intérieur de la ligne
    deux_trop_loin = ~est_3 & surement_3            # « 2 points » au-delà de la ligne
    print("  b) SHOT_TYPE vs position")
    print(f"     tirs à 3 pts situés à l'intérieur de la ligne : {trois_trop_proche.sum():,}")
    print(f"     tirs à 2 pts situés au-delà de la ligne       : {deux_trop_loin.sum():,}")
    print(f"     total incohérent : {(trois_trop_proche | deux_trop_loin).sum():,} "
          f"({(trois_trop_proche | deux_trop_loin).mean():.2%}) — conservés, SHOT_TYPE fait foi")
    return df
 
 
# ---------------------------------------------------------------- 7. arénas
def controler_arenas(df: pd.DataFrame) -> None:
    """Compare la distribution des distances par aréna (équipe locale). Ne retire rien."""
    par_arena = df.groupby("HTM").agg(
        tirs=("DIST_CALC", "size"),
        dist_moy=("DIST_CALC", "mean"),
        pct_moins_4pi=("DIST_CALC", lambda d: (d < 4).mean()),
        pct_3pts=("VALEUR_TIR", lambda v: (v == 3).mean()),
    )
    for col in ("dist_moy", "pct_moins_4pi"):
        par_arena[f"z_{col}"] = (par_arena[col] - par_arena[col].mean()) / par_arena[col].std()
 
    suspects = par_arena[(par_arena["z_dist_moy"].abs() > SEUIL_Z_ARENA)
                         | (par_arena["z_pct_moins_4pi"].abs() > SEUIL_Z_ARENA)]
 
    print(f"  Distance moyenne par aréna : de {par_arena['dist_moy'].min():.2f} "
          f"à {par_arena['dist_moy'].max():.2f} pi")
    print(f"  Arénas à plus de {SEUIL_Z_ARENA} écarts-types : {len(suspects)}")
    if len(suspects):
        print(suspects.round(3).to_string())
 
 
# ---------------------------------------------------------------- programme
def main() -> None:
    print("1. Chargement et sélection des colonnes")
    df = pd.read_parquet(SHOTS_FILE)
    noter("Extraction", len(df), len(df))
    df = garder_colonnes(df)
 
    print("\n2. Conversion des types")
    df = convertir_types(df)
 
    print("\n3. Doublons")
    df = retirer_doublons(df)
 
    print("\n4. Matchs aux rotations invalides")
    df = retirer_matchs_invalides(df)
 
    print("\n5. Tirs lointains")
    df = retirer_tirs_lointains(df)

    print("\n5b. Heaves de fin de période")
    df = retirer_heaves(df)
 
    print("\n6. Cohérence géométrique (contrôle seulement)")
    df = controler_geometrie(df)
 
    print("\n7. Biais d'aréna (contrôle seulement)")
    controler_arenas(df)
 
    print("\n8. Sauvegarde")
    df = df.drop(columns=["HTM"]).reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"  Sauvegardé : {OUT_FILE}  ({len(df):,} tirs, {df.shape[1]} colonnes)")
    print(f"  Taux de réussite final : {df['SHOT_MADE_FLAG'].mean():.3f}")
 
    print("\n=== Journal de nettoyage (pour le rapport) ===")
    print(pd.DataFrame(journal).to_string(index=False))
 
 
if __name__ == "__main__":
    main()