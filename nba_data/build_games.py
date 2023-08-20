"""Module to build the NBA games data."""
import argparse
from tqdm import tqdm
from nba_api.stats.endpoints import leaguegamefinder


def main(start_year: str, end_year: str):
    seasons_id = [
        f"{year}-{str(year+1)[-2:]}" for year in range(int(start_year), int(end_year))]

    for season_id in tqdm(seasons_id, desc="Seasons"):
        try:
            df_games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season_id,
                season_type_nullable="Regular Season"
            ).get_data_frames()[0]
            df_games.drop_duplicates(subset="GAME_ID", inplace=True)

            file_name = f"./nba_data/data/games/stats_{season_id}.csv"
            df_games.to_csv(file_name, index=False)
        except Exception as e:
            print(f"Failed for season-id: {season_id}. Reason: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to build NBA games data.")
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
