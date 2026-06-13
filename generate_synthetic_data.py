import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
NUM_MATCHES = 2000
TEAMS = ['India', 'Afghanistan']
VENUES = ['Delhi', 'Mumbai', 'Kabul', 'Dubai', 'Sharjah', 'Chennai']
TOSS_DECISIONS = ['Bat', 'Field']

# Player Mock Data
INDIA_PLAYERS = ['Rohit Sharma', 'Virat Kohli', 'Shubman Gill', 'KL Rahul', 'Suryakumar Yadav']
INDIA_BOWLERS = ['Jasprit Bumrah', 'Mohammed Shami', 'Mohammed Siraj', 'Kuldeep Yadav', 'Ravindra Jadeja']

AFG_PLAYERS = ['Rahmanullah Gurbaz', 'Ibrahim Zadran', 'Rahmat Shah', 'Hashmatullah Shahidi', 'Mohammad Nabi']
AFG_BOWLERS = ['Rashid Khan', 'Fazalhaq Farooqi', 'Mujeeb Ur Rahman', 'Naveen-ul-Haq', 'Noor Ahmad']

def generate_data(num_matches):
    data = []
    
    for i in range(num_matches):
        venue = random.choice(VENUES)
        
        # Determine Home/Away/Neutral
        if venue in ['Delhi', 'Mumbai', 'Chennai']:
            venue_type = 'Home_India'
        elif venue in ['Kabul']:
            venue_type = 'Home_Afg'
        else:
            venue_type = 'Neutral'
            
        toss_winner = random.choice(TEAMS)
        toss_decision = random.choice(TOSS_DECISIONS)
        
        # Determine who bats first
        if (toss_winner == 'India' and toss_decision == 'Bat') or (toss_winner == 'Afghanistan' and toss_decision == 'Field'):
            batting_first = 'India'
            chasing = 'Afghanistan'
        else:
            batting_first = 'Afghanistan'
            chasing = 'India'
            
        # Features
        ind_batting_strength = np.random.uniform(70, 95)
        ind_bowling_strength = np.random.uniform(75, 95)
        afg_batting_strength = np.random.uniform(60, 85)
        afg_bowling_strength = np.random.uniform(65, 88)
        
        ind_recent_form = np.random.uniform(0.5, 0.9)
        afg_recent_form = np.random.uniform(0.3, 0.7)
        
        ind_icc_rank = 1
        afg_icc_rank = 9
        
        # Player form index
        ind_player_form_idx = np.random.uniform(60, 90)
        afg_player_form_idx = np.random.uniform(50, 80)
        
        # Simulation Logic
        # India naturally has higher win probability
        win_prob_india = 0.65
        
        # Venue advantage
        if venue_type == 'Home_India':
            win_prob_india += 0.10
        elif venue_type == 'Home_Afg':
            win_prob_india -= 0.05
            
        # Form advantage
        if ind_recent_form > afg_recent_form + 0.2:
            win_prob_india += 0.05
            
        # Toss advantage
        if toss_winner == 'India':
            win_prob_india += 0.02
        else:
            win_prob_india -= 0.02
            
        # Clip probability
        win_prob_india = min(max(win_prob_india, 0.1), 0.95)
        
        winner = 'India' if random.random() < win_prob_india else 'Afghanistan'
        
        # Generate Scores
        if winner == 'India':
            if batting_first == 'India':
                ind_score = int(np.random.normal(320, 30))
                margin_runs = int(np.random.normal(50, 20))
                margin_runs = max(1, margin_runs)
                afg_score = ind_score - margin_runs
                margin = f"{margin_runs} runs"
            else:
                afg_score = int(np.random.normal(240, 30))
                ind_score = afg_score + random.randint(1, 6)
                margin_wickets = random.randint(4, 8)
                margin = f"{margin_wickets} wickets"
        else:
            if batting_first == 'Afghanistan':
                afg_score = int(np.random.normal(280, 25))
                margin_runs = int(np.random.normal(30, 15))
                margin_runs = max(1, margin_runs)
                ind_score = afg_score - margin_runs
                margin = f"{margin_runs} runs"
            else:
                ind_score = int(np.random.normal(260, 25))
                afg_score = ind_score + random.randint(1, 6)
                margin_wickets = random.randint(3, 6)
                margin = f"{margin_wickets} wickets"
                
        # Highest Scorer and Wicket Taker
        if winner == 'India':
            top_scorer = random.choices(INDIA_PLAYERS, weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0]
            top_bowler = random.choices(INDIA_BOWLERS, weights=[0.3, 0.25, 0.2, 0.15, 0.1])[0]
        else:
            top_scorer = random.choices(AFG_PLAYERS, weights=[0.3, 0.25, 0.2, 0.15, 0.1])[0]
            top_bowler = random.choices(AFG_BOWLERS, weights=[0.4, 0.2, 0.2, 0.1, 0.1])[0]
            
        # If loser team player performs exceptionally
        if random.random() < 0.2:
            top_scorer = random.choice(AFG_PLAYERS if winner == 'India' else INDIA_PLAYERS)
        if random.random() < 0.2:
            top_bowler = random.choice(AFG_BOWLERS if winner == 'India' else INDIA_BOWLERS)

        data.append({
            'venue': venue,
            'venue_type': venue_type,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'batting_first': batting_first,
            'ind_batting_strength': ind_batting_strength,
            'ind_bowling_strength': ind_bowling_strength,
            'afg_batting_strength': afg_batting_strength,
            'afg_bowling_strength': afg_bowling_strength,
            'ind_recent_form': ind_recent_form,
            'afg_recent_form': afg_recent_form,
            'ind_icc_rank': ind_icc_rank,
            'afg_icc_rank': afg_icc_rank,
            'ind_player_form_idx': ind_player_form_idx,
            'afg_player_form_idx': afg_player_form_idx,
            'ind_score': ind_score,
            'afg_score': afg_score,
            'margin': margin,
            'winner': winner,
            'top_scorer': top_scorer,
            'top_bowler': top_bowler
        })
        
    df = pd.DataFrame(data)
    
    # Calculate H2H record dynamically (dummy proxy for past matches)
    df['ind_h2h_win_pct'] = 0.80 + np.random.uniform(-0.05, 0.05, size=len(df))
    
    # Average team score
    df['ind_avg_score'] = df['ind_score'].rolling(window=10, min_periods=1).mean().fillna(280)
    df['afg_avg_score'] = df['afg_score'].rolling(window=10, min_periods=1).mean().fillna(240)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_match_data.csv', index=False)
    print(f"Generated {num_matches} matches and saved to data/synthetic_match_data.csv")

if __name__ == "__main__":
    generate_data(NUM_MATCHES)
