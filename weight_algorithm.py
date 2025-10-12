# WILL NOT BE USED

import pandas as pd

def load_teams_data():
    try:
        teams_df = pd.read_csv('dataset/teams.csv')
        return teams_df
        
    except FileNotFoundError:
        print("Error: teams.csv file not found in dataset folder!")
        return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    teams_data = load_teams_data()

    if teams_data is not None:
        print("\nData Successfully loaded.")

        clean_data = teams_data[['franchID', 'confID', 'year', 'rank']]

        weighted_ranks = {}

        for _, row in clean_data.iterrows():
            team = row['franchID'] # franchID is the ID of the current team, if a team didnt change its ID, it will be the same as tmID
            year = row['year']
            rank = row['rank']

            weight_year = 0.1 * year

            weight = (1 / rank) * weight_year

            if team in weighted_ranks:
                weighted_ranks[team] += weight
            else:
                weighted_ranks[team] = weight

        participation_counts = clean_data['franchID'].value_counts()
        for team, count in participation_counts.items():
            if team in weighted_ranks:
                weighted_ranks[team] += count * 0.1

        sorted_weighted_ranks = sorted(weighted_ranks.items(), key=lambda x: x[1], reverse=True)

        for team, weighted_rank in sorted_weighted_ranks:
            print(f"Team ID: {team}, Calculated Weight: {weighted_rank:.2f}")
