import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_teams_data():
    try:
        teams = pd.read_csv('dataset/teams.csv')
        
        return teams
        
    except FileNotFoundError:
        print("Error: teams.csv file not found in dataset folder!")
        return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_teams_post_data():
    try:
        teams_post = pd.read_csv('dataset/teams_post.csv')
        
        return teams_post
        
    except FileNotFoundError:
        print("Error: teams_post.csv file not found in dataset folder!")
        return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def load_series_data():
    try:
        series = pd.read_csv('dataset/series_post.csv')
        
        return series
        
    except FileNotFoundError:
        print("Error: series.csv file not found in dataset folder!")
        return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    teams_data = load_teams_data()
    teams_post_data = load_teams_post_data()
    series_data = load_series_data()

    if teams_data is not None and teams_post_data is not None and series_data is not None:
        print("\n Data Sucessfully loaded.")
        #print(teams_data.head())

        clean_data = teams_data[['tmID', 'confID','year','rank']]
        # print(clean_data.to_string())

        team_conf_map = teams_data.groupby('tmID')['confID'].last().to_dict()
        
        print(f"\nTeam-Conference Map:")
        print(f"Total unique teams: {len(team_conf_map)}")
        
        # Display the mapping in a nice format
        for team_id, conf_id in sorted(team_conf_map.items()):
            print(f"{team_id}: {conf_id}")
        
        # Also create a DataFrame version for easier analysis
        team_conf_df = pd.DataFrame(list(team_conf_map.items()), 
                                   columns=['tmID', 'confID'])
        print(f"\nTeam-Conference Mapping as DataFrame:")
        print(team_conf_df.to_string(index=False))
        
        # Show conference distribution
        conf_counts = team_conf_df['confID'].value_counts()
        print(f"\nTeams per Conference:")
        print(conf_counts.to_string())

        result = clean_data.groupby('tmID')['rank'].mean()
        print(f"\nTotal Rank per Team:")
        print(result.to_string())

        df = pd.DataFrame(clean_data)

        df['weighted_rank'] = df['rank'] * df['year']

        result = df.groupby('tmID')['weighted_rank'].sum().reset_index()
        print(f"\nWeighted Rank per Team:")
        print(result.to_string())

        finals = series_data[series_data['round'] == 'F']
        print(f"\nFinals Data:")
        print(finals.to_string(index=False))

        champions = finals[['year', 'tmIDWinner']].rename(columns={'tmIDWinner': 'champion_team'})
        print(f"\nChampions Data:")
        print(champions.to_string(index=False))

        teams_data = teams_data.merge(champions, on='year', how='left')
        teams_data['champion'] = (teams_data['tmID'] == teams_data['champion_team']).astype(int)

        print(teams_data[['year','tmID','champion']].sort_values(by=['champion'], ascending=False).head(10))

        feature_cols = ['rank', "o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts", "won","lost","GP","homeW","homeL","awayW","awayL","confW","confL"]
        
        X = teams_data[feature_cols]
        y = teams_data['champion']

        train = teams_data[teams_data['year'] < 9]
        test = teams_data[teams_data['year'] == 10]

        x_train, y_train = train[feature_cols], train['champion']
        x_test, y_test = test[feature_cols], test['champion']

        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(x_train, y_train)
        y_proba = model.predict_proba(x_test)[:, 1]
        test["pred_proba"] = y_proba

        pred_champions = (
        test.loc[test.groupby("year")["pred_proba"].idxmax(),
                ["year", "tmID", "pred_proba", "champion"]]
        )

        print(pred_champions)

        accuracy = pred_champions["champion"].mean()
        print(f"Champion prediction accuracy: {accuracy:.2%}")


