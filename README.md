# Unity _Reacher_ Environment

This is an implementaiton of Udacity Deep Reinforcement Learning Nanodegree's second project. In this project, the agent's goal is to maintain the position of a double-jointed arm at the target location for as many time steps as possible.. 

![](images/trained_gif.gif)

## Environment Details
This is a continuous episodic RL problem and the task is considered `solved` when the agent gets an _average score_ of at least 30 over 100 consecutive episodes.

| Space        | Type | Dim          | Description  |
| ------------- |:------:|:-------:| :-----|
| Observation      | Continuous | [33,] | Contains the agent's velocity, along with ray-based perception<br> of objects around agent's forward direction |
| Action      | Continuous | [4,] ∈ [-1, +1]    |   Contains torque applicable to two joints  |
| Reward      | Discrete | [1,]     |   +0.1 for each step that the agent's hand is in the goal location |


## Getting Started
Make sure the environment and the Python interface have compatible versions (Python 3.6). 
```bash
foo@bar:~$ python --version                                                                                      
Python 3.6.5 :: Anaconda, Inc.
```
Install the dependencies or install the project using pip. 
```bash
foo@bar:~$ pip install -r requirements.txt
```
or 
```bash
foo@bar:~$ pip install . -e
```

## Run the Agent
`main.py` is the cli to train/watch the agent. You can find the hyperparameters in `model/config.yaml`.
```bash
foo@bar:~$ python main.py -h
usage: main.py [-h] [--train]

Unity Reacher Agent

optional arguments:
  -h, --help      show this help message and exit
  --train         Train the DDPG agent if used, else load the trained weights
                  and play the game
  --weights PATH  path of .pth file with the trained weights
```
For further details run `Reacher.ipynb`.
