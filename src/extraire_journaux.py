"""
étape 1b - extraction des journaux des joueurs 2025-26 (playergamelogs)
project SMOE - un seul appel pour toute la saison.

utilisation (depuis la racine du projet, .venv activé):
    python src/extraire_journaux.py

si le json brut existe déjà, il est réutilisé (aucun nouvel appel à l'api).
"""

import json
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

# --------------------------------------------------------------- paramètres
SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
RAW_JSON = Path("data/raw/playergamelogs_2025_26.json")
OUT_FILE = Path("data/raw/playergamelogs_2025_26.parquet")
SHOTS_FILE = Path("data/raw/shots_2025_26.parquet")     # pour le contrôle de jointure
MAX_ESSAIS = 3
TIMEOUT = 60

COLONNES = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN"]


# -------------------------------------------------------------- extraction
def appeler_api() -> dict | None:
    """appelle playergamelogs pour toute la saison. renvoie le json brut ou None."""
    for essai in range(1, MAX_ESSAIS + 1):
        try:
            reponse = playergamelogs.PlayerGameLogs(
                season_nullable=SEASON,
                season_type_nullable=SEASON_TYPE,
                timeout=TIMEOUT,
            )
            return reponse.get_dict()
        except Exception as e:
            attente = 5 * essai
            print(f"  essai {essai}/{MAX_ESSAIS} échoué ({type(e).__name__}: {e}) "
                  f"- nouvelle tentative dans {attente} s")
            time.sleep(attente)
    return None


def obtenir_brut() -> dict | None:
    """réutilise le json brut s'il existe, sinon appelle l'api et le sauvegarde."""
    if RAW_JSON.exists():
        print(f"JSON brut déjà présent ({RAW_JSON}), réutilisé")
        return json.loads(RAW_JSON.read_text(encoding="utf-8"))

    print(f"appel à playergamelogs ({SEASON}, {SEASON_TYPE})...")
    brut = appeler_api()
    if brut is None:
        return None
    RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(brut), encoding="utf-8")
    print(f"JSON brut sauvegardé : {RAW_JSON}")
    return brut


def json_vers_df(brut: dict) -> pd.DataFrame:
    """extrait la table PlayergameLogs et garde les colonnes utiles."""
    result_sets = brut["resultSets"]
    table = next((rs for rs in result_sets if rs["name"] == "PlayerGameLogs"),
                  result_sets[0])
    df = pd.DataFrame(table["rowSet"], columns=table["headers"])

    df = df[COLONNES].copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)     # garder les 0 en tête
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.normalize()
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    return df


# --------------------------------------------------------------- vérifications
def verifier(logs: pd.DataFrame) -> None:
    doublons = logs.duplicated(["PLAYER_ID", "GAME_ID"]).sum()
    equipes_par_joueur = logs.groupby("PLAYER_ID")["TEAM_ID"].nunique()

    print("\n=== vérifications ===")
    print(f"lignes (joueur x match) : {len(logs):,}     (attendu ~ 25 000 à 27 000)")
    print(f"matchs distincts        : {logs['GAME_ID'].nunique()}   (attendu 1230)")
    print(f"joueurs distincts       : {logs['PLAYER_ID'].nunique()}")
    print(f"doublons (joueur, match): {doublons}    (attendu 0)")
    print(f"période                 : {logs['GAME_DATE'].min().date()} -> "
          f"{logs['GAME_DATE'].max().date()}")
    print(f"min manquantes          : {logs['MIN'].isna().sum()}")
    print(f"joueurs échangés (2+ équipes) : {(equipes_par_joueur > 1).sum()}    (normal)")

    # contrôle le plus important : chaque tir doit avoir son journal
    if not SHOTS_FILE.exists():
        print(f"\n(contrôle de jointure sauté : {SHOTS_FILE} introuvable)")
        return

    tirs = pd.read_parquet(SHOTS_FILE, columns=["GAME_ID", "PLAYER_ID"])
    tirs["GAME_ID"] = tirs["GAME_ID"].astype(str).str.zfill(10)
    tirs["PLAYER_ID"] = tirs["PLAYER_ID"].astype(int)

    couples_tirs = tirs.drop_duplicates()
    fusion = couples_tirs.merge(logs[["GAME_ID", "PLAYER_ID"]],
                                on=["GAME_ID", "PLAYER_ID"],
                                how="left", indicator=True)
    sans_journal = fusion[fusion["_merge"] == "left_only"]
    tirs_touches = tirs.merge(sans_journal[["GAME_ID", "PLAYER_ID"]],
                              on=["GAME_ID", "PLAYER_ID"]).shape[0]

    print("\n=== jointure tirs .. journaux ===")
    print(f"couples (match, joueur) dans les tirs  : {len(couples_tirs):,}")
    print(f"couples sans journal                   : {len(sans_journal)}    (attendu 0)")
    print(f"tirs concernés                         : {tirs_touches}")
    if len(sans_journal):
        print("exemples :")
        print(sans_journal[["GAME_ID", "PLAYER_ID"]].head(10).to_string(index=False))


# --------------------------------------------------------------- programme
def main() -> None:
    brut = obtenir_brut()
    if brut is None:
        print("échec de l'appel - relance le script plus tard.")
        return
    
    logs = json_vers_df(brut)
    verifier(logs)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(OUT_FILE, index=False)
    print(f"\nsauvegardé : {OUT_FILE}   ({len(logs):,} lignes)")


if __name__ == "__main__":
    main()