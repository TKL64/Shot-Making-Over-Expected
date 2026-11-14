"""
Étape 1a - Extraction des tirs de la saison 2025-26 (shotchartdetail)
project SMOE - une requête par équipe pour éviter la troncatur à 102 400 lignes.

utilisation:
    python extraire_tirsépy --test  # teste une seule équipe, sans rien combiner
    python extraire_tirs.py         # extrait les 30 équipes, puis combine et vérifie

Le script peut être relancé: les équipes déjà sauvegardées sont sautées.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

# ---------------------------------------------------------------- paramètres
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
RAW_DIR = Path("data/raw/shots")                 # un JSON brut par équipe
OUT_FILE = Path("data/raw/shots_2025_26.parquet")
PAUSE = 1.0           # secondes entre 2 appels
MAX_ESSAIS = 3        # essais par équipe avant d'abandonner
TIMEOUT = 60          # secondes
TRONCATURE = 102_400  # limite signalée de lignes par requête


# ---------------------------------------------------------------- extraction
def appeler_api(team_id: int) -> dict | None:
    """appelle shotchartdetail pour une équipe. renvoie le json brut ou None."""
    for essai in range(1, MAX_ESSAIS + 1):
        try:
            reponse = shotchartdetail.ShotChartDetail(
                team_id=team_id,
                player_id=0,                        # 0 --> tous les joueurs
                season_nullable=SEASON,
                season_type_all_star=SEASON_TYPE,
                context_measure_simple="FGA",       # pas PTS --> tirs réussis seulement
                timeout=TIMEOUT
            )
            return reponse.get_dict()
        except Exception as e:  # réseau, délai dépassé, réponse invalide...
            attente = 5 * essai
            print(f"    essai {essai}/{MAX_ESSAIS} échoué ({type(e).__name__}: {e}) "
                  f"- nouvelle tentative dans {attente} s")
            time.sleep(attente)
    return None


def json_vers_df(brut: dict) -> pd.DataFrame:
    """extrait la table des tirs (Shot_chart_Detail) du json brut."""
    result_sets = brut["resultSets"]
    table = next((rs for rs in result_sets if rs["name"] == "Shot_Chart_Detail"),
                  result_sets[0])
    df = pd.DataFrame(table["rowSet"], columns=table["headers"])
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)  # garder les 0 en tête
    return df


def extraire_equipe(equipe: dict) -> Path | None:
    """extrait et sauvegarde un équipe. saute l'équipe si le fichier existe."""
    fichier = RAW_DIR / f"shots_{equipe['id']}.json"
    if fichier.exists():
        print(f"  {equipe['abbreviation']:>3} - déjà présent, sauté")
        return fichier

    brut = appeler_api(equipe["id"])
    if brut is None:
        print(f"  {equipe['abbreviation']:>3} - ÉCHEC, relance le script plus tard")
        return None

    fichier.write_text(json.dumps(brut), encoding="utf-8")
    n = len(json_vers_df(brut))
    alerte = "   !!! TRONQUÉ ?" if n >= TRONCATURE else ""
    print(f"   {equipe['abbreviation']:>3} - {n:>5} tirs sauvegardés{alerte}")
    time.sleep(PAUSE)
    return fichier


# ---------------------------------------------------------------- vérifications
def verifier(df: pd.DataFrame, n_equipes_attendu: int) -> None:
    dates = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d")
    doublons = df.duplicated(["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"]).sum()
    taux = df["SHOT_MADE_FLAG"].astype(int).mean()

    print("\n=== vérifications ===")
    print(f"tirs au total       : {len(df):,}   (attendu ~ 220 000 pour 30 équipes)")
    print(f"équipes             : {df['TEAM_ID'].nunique()}     (attendu {n_equipes_attendu})")
    print(f"matchs distincts    : {df['GAME_ID'].nunique()}     (attendu ~ 1230 pour 30 équipes)")
    print(f"doublons            : {doublons}   (attendu 0)")
    print(f"taux de réussite    : {taux:.3f}   (attendu ~ 0.46 ; 1,000 = 'FGA' oublié)")
    print(f"période             : {dates.min().date()} -> {dates.max().date()}")

    par_equipe = df.groupby("TEAM_ID").size()
    print(f"tirs par équipe       : min {par_equipe.min()}, {par_equipe.max()}")
    if (par_equipe >= TRONCATURE).any():
        print("!!! au moins une équipe a atteint la limite de troncature!")


# --------------------------------------------------------------- programme
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="extraire une seule équipe pour tester")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    equipes = sorted(teams.get_teams(), key=lambda t: t["abbreviation"])
    if args.test:
        equipes = equipes[:1]

    print(f"extraction {SEASON} ({SEASON_TYPE}) - {len(equipes)} équipe(s)")
    fichiers = [f for f in (extraire_equipe(e) for e in equipes) if f is not None]

    manquantes = len(equipes) - len(fichiers)
    if manquantes:
        print(f"\n{manquantes} équipe(s) manquante(s) - relance le script pour les récupérer.")
        return

    df = pd.concat(
        [json_vers_df(json.loads(f.read_text(encoding="utf-8"))) for f in fichiers],
        ignore_index=True
    )
    verifier(df, n_equipes_attendu=len(equipes))

    if not args.test:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_FILE, index=False)
        print(f"\nsauvegardé : {OUT_FILE}   ({len(df):,} lignes)")


if __name__ == "__main__":
    main()