# Giotto AI

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) 

This repository contains the env implementation of the two player grid based games TicTacToe and Connect4, together with the Pygame implementation of the games, and it features a variety of agents that can play the game. It's possible to play against the agents or make them challenge eachother. 

## Games
Currently implemented games are:
- Tic Tac Toe (Tris)
- Connect 4
The code is structured to make it easy to expand the selection of games with others similarly structured, such as Connect 5 or Ultimate TicTacToe.

## Agents
Currently implemented agents are:
- Human
- Random
- Minimax
- Monte Carlo Tree Search

## Structure
The repository is structured as follows:
- In the `giotto/envs` folder you can find the game environments. These define the games rules and logic.
- In the `giotto/games` folder are the pygame interfaces for the game environments.
- In the `giotto/scripts` are utilities to play in text mode or in Pygame, as well as simulate many games between agents or evaluate their play from a fixed/random starting move. 

## Installation
The repository is setup us as a poetry project and by default requires Python 3.10 or later.
To install the repository you can follow these steps:

First, install `poetry` if you haven't already, as indicated by the instructions on the [Poetry installation page](https://python-poetry.org/docs/).
Then, clone the repository to your local machine using the following command:
```
git clone https://github.com/nicolapesaresi/Giotto
cd Giotto
```
Create and activate a virtual environment:
```
# replace `myenv` with the name of your virtual environment
python3 -m venv myenv
source myenv/bin/activate
```
Or, if using `Conda`:
```
conda create -n myenv python=3.10
conda activate myenv
```
Use Poetry to install the project dependencies:
```
poetry install
```
Now you can launch the game on desktop with:
```
python giotto/scripts/run_pygame.py
```

## Resources
[Pygame documentation](https://www.pygame.org/docs/)
[Minimax algorithm](https://en.wikipedia.org/wiki/Minimax)
[Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
[Tic Tac Toe](https://en.wikipedia.org/wiki/Tic-tac-toe)
[Connect 4](https://en.wikipedia.org/wiki/Connect_Four)