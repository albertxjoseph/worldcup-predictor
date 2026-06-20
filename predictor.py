import pandas as pd

# this is the address of a free dataset of international match results
url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# read the data from that address into a table called df
df = pd.read_csv(url)

# show how big the table is, and the last few rows
print(df.shape)
print(df.tail())

# make sure scores exist and sort matches from oldest to newest
df = df.dropna(subset=["home_score", "away_score"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# a dictionary to hold each team's rating. everyone starts at 1500.
ratings = {}

# this formula turns a rating gap into a win chance between 0 and 1
def expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# go through every match in order and update the two teams' ratings
for game in df.itertuples():
    home = ratings.get(game.home_team, 1500)
    away = ratings.get(game.away_team, 1500)

    home_boost = 0 if game.neutral else 65   # playing at home is an advantage
    chance_home = expected(home + home_boost, away)

    if game.home_score > game.away_score:
        actual = 1            # home won
    elif game.home_score < game.away_score:
        actual = 0            # home lost
    else:
        actual = 0.5          # draw

    change = 30 * (actual - chance_home)
    ratings[game.home_team] = home + change
    ratings[game.away_team] = away - change

# show the ten highest rated teams
top_teams = sorted(ratings.items(), key=lambda x: -x[1])[:10]
for team, rating in top_teams:
    print(team, round(rating))

def predict(team_a, team_b):
    rating_a = ratings.get(team_a, 1500)
    rating_b = ratings.get(team_b, 1500)
    chance_a = expected(rating_a, rating_b)
    print(team_a, "win chance:", round(chance_a * 100), "%")
    print(team_b, "win chance:", round((1 - chance_a) * 100), "%")

# try it out, change the names to any two countries
predict("Austria", "Jordan")