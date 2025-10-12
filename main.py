# NOT BEING USED

import pandas as pd

df = pd.read_csv('./dataset/teams.csv')

#Just put some random numbers now
weight_array = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

def custom_score(team_df):
    
    inv_rank = 1 / team_df["rank"]

    weights = team_df["year"].apply(lambda y: weight_array[y])

    seasons_played = team_df["year"].nunique()

    team_score = ((inv_rank * weights) * (0.1 * seasons_played)).mean()

    return team_score

df_east = df[df["confID"] == "EA"]
df_west = df[df["confID"] == "WE"]

east_scores = df_east.groupby("tmID").apply(custom_score).reset_index()
east_scores.columns = ["tmID", "team_score"]
east_scores = east_scores.sort_values(by="team_score", ascending=False)

west_scores = df_west.groupby("tmID").apply(custom_score).reset_index()
west_scores.columns = ["tmID", "team_score"]
west_scores = west_scores.sort_values(by="team_score", ascending=False)

print("East Conference Rankings:\n", east_scores)
print("\nWest Conference Rankings:\n", west_scores)
