import json
import pandas as pd
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import atexit

app = Flask(__name__)

matplotlib.use('Agg')

# Load the match data from CSV
match_data = pd.read_csv('filtered_data.csv')

# Load the ONNX model
onnx_model_path = "RNN.onnx"
session = ort.InferenceSession(onnx_model_path)

# Mapping of team names to their possible variations
team_name_mapping = {
    "Arsenal": ["Arsenal"],
    "Aston Villa": ["Aston Villa"],
    "Bournemouth": ["Bournemouth"],
    "Brentford": ["Brentford"],
    "Brighton": ["Brighton", "Brighton & Hove Albion"],
    "Chelsea": ["Chelsea"],
    "Crystal Palace": ["Crystal Palace"],
    "Everton": ["Everton"],
    "Fulham": ["Fulham"],
    "Ipswich": ["Ipswich Town", "Ipswich"],
    "Leicester": ["Leicester City", "Leicester"],
    "Liverpool": ["Liverpool"],
    "Man City": ["Manchester City", "Man City"],
    "Man United": ["Manchester United", "Man United"],
    "Newcastle": ["Newcastle United", "Newcastle"],
    "Nottingham": ["Nottingham Forest", "Nottingham"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham"],
    "West Ham": ["West Ham United", "West Ham"],
    "Wolves": ["Wolverhampton Wanderers", "Wolves"]
}

def normalize_team_name(team_name):
    suffixes = ['FC', 'F.C.', 'Football Club', 'F.C', 'FC']
    for suffix in suffixes:
        if team_name.endswith(suffix):
            team_name = team_name[:-len(suffix)].strip()

    for standard_name, variations in team_name_mapping.items():
        if team_name in variations:
            return standard_name
    return team_name

def load_fixtures():
    with open('en.1.json', 'r') as file:
        data = json.load(file)
        matches = data['matches']
        
        filtered_matches = []
        for match in matches:
            filtered_match = {
                'round': match.get('round', 'Unknown Round'),
                'date': match.get('date', 'Unknown Date'),
                'home_team': normalize_team_name(match.get('team1', 'Unknown Team')),
                'away_team': normalize_team_name(match.get('team2', 'Unknown Team'))
            }
            filtered_matches.append(filtered_match)
        return filtered_matches

def get_team_strength(match_data, team_name):
    normalized_name = normalize_team_name(team_name)
    match_data['NormalizedHomeTeam'] = match_data['HomeTeam'].apply(normalize_team_name)
    match_data['NormalizedAwayTeam'] = match_data['AwayTeam'].apply(normalize_team_name)

    filtered_row = match_data[(
        match_data['NormalizedHomeTeam'] == normalized_name) | 
        (match_data['NormalizedAwayTeam'] == normalized_name)]
    
    if filtered_row.empty:
        raise ValueError(f"No data found for team: {team_name}")

    row = filtered_row.iloc[-1]

    home_goals = row['Sum of FullTimeHomeTeamGoals']
    away_goals = row['Sum of FullTimeAwayTeamGoals']
    home_streak = row.get('home_team_streak', 0)
    away_streak = row.get('away_team_streak', 0)

    return np.array([home_goals, away_goals, home_streak], dtype=np.float32)

def get_last_n_matches(team_name, n=5):
    normalized_name = normalize_team_name(team_name)
    filtered_matches = match_data[
        (match_data['HomeTeam'] == normalized_name) | 
        (match_data['AwayTeam'] == normalized_name)
    ].tail(n)
    
    return filtered_matches

def calculate_goals_and_assists(filtered_matches, team_name):
    goals = assists = 0
    for _, row in filtered_matches.iterrows():
        if row['HomeTeam'] == team_name:
            goals += row['Sum of FullTimeHomeTeamGoals']
            assists += row.get('HomeAssists', 0)
        else:
            goals += row['Sum of FullTimeAwayTeamGoals']
            assists += row.get('AwayAssists', 0)
    return goals, assists

def create_pie_chart(winning_probs, home_team, away_team):
    labels = [home_team, away_team, 'Draw']
    win1, win2 = winning_probs
    draw_prob = 1 - (win1 + win2)

    # Ensure probabilities sum to 1
    sizes = [win1, win2, draw_prob]

    # Colors and explosion settings
    colors = ['#FF6384', '#36A2EB', '#FFCE56']
    explode = (0.1, 0, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def predict_winner(home_team, away_team):
    home_strength = get_team_strength(match_data, home_team)
    away_strength = get_team_strength(match_data, away_team)

    features = np.array([[home_strength, away_strength]], dtype=np.float32).reshape(1, 1, 6)
    
    inputs = {session.get_inputs()[0].name: features}
    outputs = session.run(None, inputs)

    probabilities = outputs[0][0]
    win1 = probabilities[0]
    win2 = probabilities[1] if len(probabilities) > 1 else 1 - win1

    # Return both winning probabilities and calculate the draw
    return np.array([win1, win2])

def get_last_meetings(match_data, home_team, away_team, n=5):
    """Fetch the last n meetings between two clubs and return the results."""
    normalized_home = normalize_team_name(home_team)
    normalized_away = normalize_team_name(away_team)
    
    filtered_matches = match_data[
        ((match_data['HomeTeam'] == normalized_home) & (match_data['AwayTeam'] == normalized_away)) |
        ((match_data['HomeTeam'] == normalized_away) & (match_data['AwayTeam'] == normalized_home))
    ].tail(n)

    if filtered_matches.empty:
        return None  # No matches found

    results = filtered_matches['FullTimeResult'].value_counts()
    home_wins = results.get('H', 0)
    away_wins = results.get('A', 0)
    draws = results.get('D', 0)
    
    return home_wins, away_wins, draws

def create_meeting_pie_chart(home_wins, away_wins, draws):
    """Create a pie chart for the results of the last meetings."""
    labels = ['Home Wins', 'Away Wins', 'Draws']
    sizes = [home_wins, away_wins, draws]

    if sum(sizes) == 0:
        return None  # No games played

    colors = ['gold', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0)  # explode the first slice (Home Wins)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free up memory
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/football')
def football():
    matches = load_fixtures()
    return render_template('football.html', matches=matches)

@app.route('/match_details/<int:match_id>')
def match(match_id):
    matches = load_fixtures()
    selected_match = matches[match_id - 1]

    # Predict winner and calculate probabilities
    prediction = predict_winner(selected_match['home_team'], selected_match['away_team'])

    # Get last matches data
    home_last_matches = get_last_n_matches(selected_match['home_team'])
    away_last_matches = get_last_n_matches(selected_match['away_team'])

    # Calculate goals and assists
    home_goals, home_assists = calculate_goals_and_assists(home_last_matches, selected_match['home_team'])
    away_goals, away_assists = calculate_goals_and_assists(away_last_matches, selected_match['away_team'])

    # Get last meetings between two teams
    home_wins, away_wins, draws = get_last_meetings(match_data, selected_match['home_team'], selected_match['away_team'])
    
    # Create pie chart for last meetings
    if home_wins is None:
        meeting_pie_chart = None  # No matches found
    else:
        meeting_pie_chart = create_meeting_pie_chart(home_wins, away_wins, draws)

    # Create pie chart with team names and probabilities
    pie_chart = create_pie_chart(prediction, selected_match['home_team'], selected_match['away_team'])

    return render_template('match_details.html', 
                           match=selected_match, 
                           prediction=prediction, 
                           home_goals=home_goals, 
                           home_assists=home_assists, 
                           away_goals=away_goals, 
                           away_assists=away_assists, 
                           pie_chart=pie_chart,
                           meeting_pie_chart=meeting_pie_chart)

@atexit.register
def cleanup():
    plt.close('all')  # Close all Matplotlib figures

if __name__ == '__main__':
    app.run(debug=True)
