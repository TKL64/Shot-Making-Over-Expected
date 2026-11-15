"""
étape 1c (suite) diagnostique des rotations : trouver les matchs aux données invalides.
projet SMOE

utilisation (depuis la racine du projet, .venv activé) :
    python src/diagnostiquer_rotations.py               # liste les matchs problématiques
    python src/diagnostiquer_rotations.py --supprimer   # supprime aussi leur json bruts
                                                        # pour les retélécharger avec
                                                        # extraire_rotations.py

un match est jugé invalide s'il remplit au moins un de ces critères :
    1. un séjour a une durée négative (OUT_TIME_REAL < IN_TIME_REAL) ;
    2. le total d'une équipe s'écarte de 5 x durée du match de plus de 2 minutes ;
    3. un joueur a joué selon son journal (MIN > 0), mais est absent des rotations ;
    4. un joueur a un écart de plus de 2 minutes entre ses séjours et MIN du journal.
"""

import argparse
from pathlib import Path

import pandas as pd

ROT_FILE = Path("data/raw/rotations_2025_26.parquet")
LOGS_FILE = Path("data/raw/playergamelogs_2025_26.parquet")
SHOTS_FILE = Path("data/raw/shots_2025_26.parquet")
RAW_DIR = Path("data/raw/rotations")
OUT_CSV = Path("data/raw/rotations_matchs_invalides.csv")

DIXIEMES_PAR_MIN = 600      # unité confirmée : dixièmes de seconde
TOLERANCE_MIN = 2.0


def charger() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rot = pd.read_parquet(ROT_FILE)
    logs = pd.read_parquet(LOGS_FILE)
    tirs = pd.read_parquet(SHOTS_FILE, columns=["GAME_ID", "PLAYER_ID"])
    for df in (rot, logs, tirs):
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    rot["DUREE_MIN"] = (rot["OUT_TIME_REAL"] - rot["IN_TIME_REAL"]) / DIXIEMES_PAR_MIN
    return rot, logs, tirs


def diagnostiquer(rot: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """renvoie une ligne par match invalide avec les critères qui échouent."""
    raisons: dict[str, set[str]] = {}

    def marquer(game_ids, raison):
        for g in game_ids:
            raisons.setdefault(g, set()).add(raison)

    # 1. durée négatives (les durée nulles sont sans danger et ignorées)
    marquer(rot.loc[rot["DUREE_MIN"] < 0, "GAME_ID"].unique(), "duree_negative")

    # 2. total par équipe vs 5 * durée du match (durée = OUT_TIME_REAL max du match)
    duree_match = rot.groupby("GAME_ID")["OUT_TIME_REAL"].max() / DIXIEMES_PAR_MIN
    total_equipe = rot.groupby(["GAME_ID", "TEAM_ID"])["DUREE_MIN"].sum().reset_index()
    total_equipe["ATTENDU"] = 5 * total_equipe["GAME_ID"].map(duree_match)
    ecart_eq = (total_equipe["DUREE_MIN"] - total_equipe["ATTENDU"]).abs()
    marquer(total_equipe.loc[ecart_eq > TOLERANCE_MIN, "GAME_ID"].unique(), "total_equipe")

    # 3. et 4. comparaison joueur par joueur avec le journal
    par_joueur = (rot.groupby(["GAME_ID", "PERSON_ID"])["DUREE_MIN"].sum()
                  .reset_index().rename(columns={"PERSON_ID": "PLAYER_ID"}))
    comp = logs.loc[logs["MIN"] > 0, ["GAME_ID", "PLAYER_ID", "MIN"]].merge(
        par_joueur, on=["GAME_ID", "PLAYER_ID"], how="left")
    marquer(comp.loc[comp["DUREE_MIN"].isna(), "GAME_ID"].unique(), "joueur_absent")
    ecart_j = (comp["DUREE_MIN"] - comp["MIN"]).abs()
    marquer(comp.loc[ecart_j > TOLERANCE_MIN, "GAME_ID"].unique(), "ecart_joueur")

    return pd.DataFrame(
        [{"GAME_ID": g, "RAISONS": ", ".join(sorted(r))} for g, r in sorted(raisons.items())],
        columns=["GAME_ID", "RAISONS"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supprimer", action="store_true",
                        help="supprimer les json bruts des matchs invalides")
    args = parser.parse_args()

    rot, logs, tirs = charger()
    invalides = diagnostiquer(rot, logs)

    tirs_touches = tirs["GAME_ID"].isin(invalides["GAME_ID"]).sum()
    print("=== diagnostique des rotations ===")
    print(f"matchs invalides : {len(invalides)} / {rot['GAME_ID'].nunique()}")
    print(f"tirs dans ces matchs : {tirs_touches:,} / {len(tirs):,}"
          f"({tirs_touches / len(tirs):.2%})")
    if len(invalides):
        print("\ndétail :")
        print(invalides.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    invalides.to_csv(OUT_CSV, index=False)
    print(f"\nliste sauvegardée : {OUT_CSV}")

    if args.supprimer and len(invalides):
        n = 0
        for g in invalides["GAME_ID"]:
            f = RAW_DIR / f"rot_{g}.json"
            if f.exists():
                f.unlink()
                n += 1
        print(f"{n} json supprimés. relance python src/extraire_rotations.py")


if __name__ == "__main__":
    main()