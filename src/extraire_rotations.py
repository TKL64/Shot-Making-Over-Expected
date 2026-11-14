"""
étape 1c - extraction des rotations 2025-26 (gamerotation)
projet SMOE - un appel par match (~1230), pour reconstruire les minutes jouées.

utilisation (depuis la racine du projet, .venv activé) :
    python src/extraire_rotations.py --test     # 5 matchs seulements, sans sauvegard finale
    python src/extraire_rotations.py            # tous les matchs, puis combinaison et vérifications

le script peut être relancé : les matchs déjà sauvgardés sont sautés.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import gamerotation

# --------------------------------------------------------------- paramètres
LOGS_FILE = Path("data/raw/playergamelogs_2025_26.parquet")   # source des GAME_ID
RAW_DIR = Path("data/raw/rotations")                          # un json brut par match
OUT_FILE = Path("data/raw/rotations_2025_26.parquet")

PAUSE = 1.0             # secondes entre deux appels
PAUSE_LONGUE = 30       # pause supplémentaire...
TOUS_LES_N = 100        # ...tous les 100 appels, pour ménager l'api
MAX_ESSAIS = 3          # essais par match
MAX_ECHECS_SUIVIS = 5   # arrêt si autant de matchs échouent d'affilée (api bloquée)
TIMEOUT = 60
N_TEST = 5

TABLES = ("HomeTeam", "AwayTeam")
COLONNES = ["GAME_ID", "TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]


# ---------------------------------------------------------------- extraction
def appeler_api(game_id: str) -> dict | None:
    """appelle gamerotation pour un match. renvoie le json brut ou None."""
    for essai in range(1, MAX_ESSAIS + 1):
        try:
            reponse = gamerotation.GameRotation(game_id=game_id, timeout=TIMEOUT)
            return reponse.get_dict()
        except Exception as e:
            attente = 5 * essai
            print(f"    essai {essai}/{MAX_ESSAIS} échoué ({type(e).__name__}: {e}) "
                  f"- nouvelle tentative dans {attente} s")
            time.sleep(attente)
    return None


def json_vers_df(brut: dict) -> pd.DataFrame:
    """combine les tables Hometeam et AwayTeam et garde les colonnes utiles."""
    morceaux = []
    for rs in brut["resultSets"]:
        if rs["name"] in TABLES and rs["rowSet"]:
            morceaux.append(pd.DataFrame(rs["rowSet"], columns=rs["headers"]))
    if not morceaux:
        return pd.DataFrame(columns=COLONNES)

    df = pd.concat(morceaux, ignore_index=True)[COLONNES].copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df["PERSON_ID"] = df["PERSON_ID"].astype(int)
    df["IN_TIME_REAL"] = pd.to_numeric(df["IN_TIME_REAL"], errors="coerce")
    df["OUT_TIME_REAL"] = pd.to_numeric(df["OUT_TIME_REAL"], errors="coerce")
    return df


def extraire_match(game_id: str) -> str:
    """extrait un match. renvoie 'present, 'ok', 'vide' ou 'echec'."""
    fichier = RAW_DIR / f"rot_{game_id}.json"
    if fichier.exists():
        return "present"

    brut = appeler_api(game_id)
    if brut is None:
        return "echec"
    if json_vers_df(brut).empty:
        return "vide"       # pas sauvegardé : sera redemandé à la prochaine exécution

    fichier.write_text(json.dumps(brut), encoding="utf-8")
    return "ok"


# ------------------------------------------------------------------- vérifications
def verifier(rot: pd.DataFrame, logs: pd.DataFrame) -> None:
    rot = rot.copy()
    rot["DUREE"] = rot["OUT_TIME_REAL"] - rot["IN_TIME_REAL"]

    print("\n=== vérifications ===")
    print(f"séjours (lignes)            : {len(rot):,}")
    print(f"matchs distincts            : {rot['GAME_ID'].nunique()}    "
          f"(attendu {logs['GAME_ID'].nunique()})")
    print(f"temps manquants (IN/OUT)    : "
          f"{rot[['IN_TIME_REAL', 'OUT_TIME_REAL']].isna().any(axis=1).sum()}   (attendu 0)")
    print(f"séjours de durée <= 0        : {(rot['DUREE'] <= 0).sum()}   (attendu 0 ou presque)")
    print(f"OUT_TIME_REAL maximum       : {rot['OUT_TIME_REAL'].max():,.0f}   "
          f"(28 800 = 48 min en dixièmes de s ; plus = prolongation)")

    # 1) unité : la somme des séjours doit correspondre à MIN du journal
    par_joueur = (rot.groupby(["GAME_ID", "PERSON_ID"])["DUREE"].sum()
                  .reset_index().rename(columns={"PERSON_ID": "PLAYER_ID"}))
    comp = par_joueur.merge(logs[["GAME_ID", "PLAYER_ID", "MIN"]],
                            on=["GAME_ID", "PLAYER_ID"], how="inner")
    ecart_dix = (comp["DUREE"] / 600 - comp["MIN"]).abs()   # 0.1 de secondes
    ecart_sec = (comp["DUREE"] / 60 - comp["MIN"]).abs()    # secondes

    print("\n=== contrôle de unités (séjours reconstruits vs MIN du journal) ===")
    print(f"si dixièmes de seconde : écart médian {ecart_dix.median():.2f} min, "
          f"{(ecart_dix <= 1).mean():.1%} à 1 min près")
    print(f"si secondes            : écart médian {ecart_sec.median():.2f} min, "
          f"{(ecart_sec <= 1).mean():.1%} à 1 min près")
    unite = "dixièmes de seconde" if ecart_dix.median() < ecart_sec.median() else "secondes"
    print(f"-> unité retenue : {unite}")

    gros = comp[(ecart_dix if unite.startswith("dix") else ecart_sec) > 2]
    print(f"couples avec un écart > 2 min : {len(gros)}")
    if len(gros):
        print(gros.head(10).to_string(index=False))

    # 2) total par équipe : 5 joueurs x durée du match (240 min en temps réglementaire)
    diviseur = 600 if unite.startswith("dix") else 60
    par_equipe = rot.groupby(["GAME_ID", "TEAM_ID"])["DUREE"].sum() / diviseur
    print(f"\nminutes par équipe et match : min {par_equipe.min():.1f}, "
          f"médiane {par_equipe.median():.1f}, max {par_equipe.max():.1f}   "
          f"(attendu 240 ; 265, 290... avec prolongation)")
    
    # 3) couverture : joueurs ayant joué (MIN > 0) mais absents des rotations
    joues = logs.loc[logs["MIN"] > 0, ["GAME_ID", "PLAYER_ID"]]
    couverts = par_joueur[["GAME_ID", "PLAYER_ID"]]
    manque = joues.merge(couverts, how="left", indicator=True)
    manque = manque[manque["_merge"] == "left_only"]
    print(f"\njoueurs du journal (min > 0) absents des rotations : {len(manque)}    (attendu 0)")
    if len(manque):
        print(manque[["GAME_ID", "PLAYER_ID"]].head(10).to_string(index=False))


# ----------------------------------------------------------------- programme
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help=f"extraire seulement {N_TEST} matchs pour tester")
    args = parser.parse_args()

    if not LOGS_FILE.exists():
        print(f"{LOGS_FILE} introuvable - fais d'abord l'étape 1b.")
        return

    logs = pd.read_parquet(LOGS_FILE)
    logs["GAME_ID"] = logs["GAME_ID"].astype(str).str.zfill(10)
    game_ids = sorted(logs["GAME_ID"].unique())
    if args.test:
        game_ids = game_ids[:N_TEST]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"extraction des rotations - {len(game_ids)} match(s)")

    bilan = {"present": 0, "ok": 0, "vide": 0, "echec": 0}
    a_refaire = []
    appels = 0
    echecs_suivis = 0

    for i, gid in enumerate(game_ids, start=1):
        statut = extraire_match(gid)
        bilan[statut] += 1

        if statut == "present":
            continue

        appels += 1
        print(f"  [{i:>4}/{len(game_ids)}] {gid} - {statut}")

        if statut in ("vide", "echec"):
            a_refaire.append(gid)
            echecs_suivis += 1
            if echecs_suivis >= MAX_ECHECS_SUIVIS:
                print(f"\n{MAX_ECHECS_SUIVIS} échecs d'affilée : l'api bloque probablement. "
                      "attend 15-30 minutes, puis relance le script.")
                break
        else:
            echecs_suivis = 0
            
        time.sleep(PAUSE)
        if appels % TOUS_LES_N == 0:
            print(f" ... pause de {PAUSE_LONGUE} s après {appels} appels")
            time.sleep(PAUSE_LONGUE)

    print(f"\nbilan : {bilan['ok']} nouveaux, {bilan['present']} déjà présents, "
          f"{bilan['vide']} vides, {bilan['echec']} échecs")

    fichiers = [RAW_DIR / f"rot_{g}.json" for g in game_ids]
    manquants = [f for f in fichiers if not f.exists()]
    if manquants:
        print(f"{len(manquants)} match(s) manquant(s) - relance le script pour les récupérer.")
        if a_refaire:
            print("exemples :", ", ".join(a_refaire[:10]))
        return

    rot = pd.concat(
        [json_vers_df(json.loads(f.read_text(encoding="utf-8"))) for f in fichiers],
        ignore_index=True
    )
    verifier(rot, logs[logs["GAME_ID"].isin(game_ids)])

    if not args.test:
        rot.to_parquet(OUT_FILE, index=False)
        print(f"\nsauvegardé : {OUT_FILE}   ({len(rot):,} lignes)")

if __name__ == "__main__":
    main()