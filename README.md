# Dice Game: Dertigen

Dertigen is a dice-based game implemented in Python. It supports multiple strategies, including a Q-learning agent for reinforcement learning. The game can be played interactively or run as a simulation to analyze strategies and outcomes.

## Features

- **Player Strategies**:
  - `RiskTaker`: Takes risks by choosing lower dice values.
  - `RiskAverse`: Plays conservatively by choosing higher dice values.
  - `SmartRiskTaker`: Uses memoized probabilities and expected values to make decisions.
  - `QLearner`: A reinforcement learning agent that learns optimal strategies through Q-learning.
  - `playSelf`: Allows a human player to make decisions interactively.

- **Simulation Mode**:
  - Run thousands of game rounds to analyze player strategies.
  - Collect statistics such as given/taken ratios, doubling success rates, and more.

- **Interactive Mode**:
  - Play the game with a graphical user interface (GUI) built using `pygame`.
  - Includes animations, sliders for adjusting game parameters, and a scoreboard.

- **Statistics and Visualization**:
  - Track and display statistics for each player, such as:
    - Times drunk per round.
    - Stripes given per round.
    - Doubling success rates.
    - Best doubling amount and extra stripes above 30.
  - Visualize dice choice frequencies and ending sum distributions using bar plots.

## Requirements

- Python 3.8 or higher
- Required libraries:
  - `pygame`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tqdm`

Install the dependencies using:

```bash
pip install pygame numpy matplotlib seaborn tqdm
