import pandas as pd

df = pd.read_csv('./dataset/teams.csv')

df_prediction = df.groupby("tmID", as_index=False)["rank"].mean()
df_sorted = df_prediction.sort_values(by="rank", ascending=False)

print(df_sorted.to_string(index=False))
