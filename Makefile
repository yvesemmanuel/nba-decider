start_year := 2019
end_year := 2024

build-games:
	python ./nba_data/build_games.py --start_year $(start_year) --end_year $(end_year)

build-players:
	python ./nba_data/build_players.py --start_year=$(start_year) --end_year=$(end_year)

build-teams:
	python ./nba_data/build_teams.py --start_year=$(start_year) --end_year=$(end_year)

build-all:
	install
	build-games
	build-players
	build-teams

install:
	pip install -r requirements.txt
