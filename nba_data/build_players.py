"""Module to build the NBA players data."""
import argparse
from tqdm import tqdm
from nba_api.stats.endpoints import PlayerGameLogs


def main(start_year: str, end_year: str):
    seasons_id = [
        f"{year}-{str(year+1)[-2:]}" for year in range(int(start_year), int(end_year))]

    for season_id in tqdm(seasons_id, desc="Seasons"):
        try:
            df_players = PlayerGameLogs(
                season_nullable=season_id,
                season_type_nullable="Regular Season",
                per_mode_simple_nullable="PerGame"
            ).get_data_frames()[0]

            file_name = f"./nba_data/data/players/stats_{season_id}.csv"
            df_players.to_csv(file_name, index=False)
        except Exception as e:
            print(f"Failed for season-id: {season_id}. Reason: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to build NBA players data.")
    parser.add_argument(
        "--start_year",
        help="The NBA start season year...",
        required=True)
    parser.add_argument(
        "--end_year",
        help="The NBA last season year...",
        required=True)

    args = parser.parse_args()
    main(args.start_year, args.end_year)
