# TurtlePlayer

TurtlePlayer is a reinforcement learning framework designed for financial trading strategies using the Turtle Trading system. It differs from most RL traders in that the action space isn't correlated with buying, selling, holding actions. Instead, the action space adjusts the lookback period for entries and exits. The 
type 1 turtle strategy enters when a close exceeds the previous 20 day's high and an exit when the close prie is below the previous 10 day's low. Turtle Player can dynamically adjust these periods(well just the entry period as in this model the exit period = entry period / 2). You can mess around with your own turtle players and use the config to generate unique turtle players. 

## Description

At it's core Turtle Player was designed to experiment with RL in trading where the action space isn't associated with buying or selling. As the turtle trading strategy and basically all variations of it have been priced in, it is highly unlikely that turtle player will be able to generate competitive retuns, but the concept of non-trading action spaces might spin out some interesting algorithms.

Turtle Player is built using Gymnasium and PyTorch for Rl and NN training. Pandas and Numpy for data loading and manipulation, and Matplotlib and tabulate for analysis.

## Installation Guide

Follow these steps to get TurtlePlayer up and running on your system:

I recommend using Anaconda here so you can avoid libary conflicts

```bash
conda create -n turtle python=3.12
```

### Step 1: Clone the Repository

Clone the TurtlePlayer repository to your local machine using the following command:

```bash
git clone https://github.com/lordyabu/TurtlePlayer.git
```

### Step 2: Navigate to TurtlePlayer directory

```bash
cd TurtlePlayer
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```


### Step 4: cd int source code

```bash
cd src
```

## Configuration and running guide

Be sure to read and configure config.py !!!!!

I've developed two agents: 1. BaseAgent(Base turtle trading algorithm) 2. DQNAgent

to run agents after modifying config.py

```bash
python run_agent.py --agent BaseAgent
python run_agent.py --agent DQNAgent
```

## Analyzing results

To analyze a specifc log number, first go into the logs folder and find the session number and episode numbers you want to look at. As there a lot of different analyzation types I won't show all the commands.

```bash
python analyze.py --type state --session1 1 --episode_nums1 1
python analyze.py --type state --session1 1 --episode_nums1 1 --session2 2 --epiosde_nums 1,2,3
python analyze.py --type trade --session1 2
python analyze.py --type train --session1 2
python analyze.py --type performance --session1 1 --session2 2 # For this a Base agent session must always be fist
```


## Actual results


### AMZN Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | 1.24%         |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 0.45%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | -0.08%        |    20    |
| Base        | 2018-03-09   | 2019-03-08 | -0.02%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | 0.13%         |    32.79 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 0.30%         |    31.15 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -0.10%        |    29.33 |
| DQN_Average | 2018-03-09   | 2019-03-08 | -0.10%        |    30.01 |


### COKE Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | 0.22%         |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 0.12%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | 0.67%         |    20    |
| Base        | 2018-03-09   | 2019-03-08 | 0.48%         |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | 1.16%         |    30.2  |
| DQN_Average | 2010-02-05   | 2011-02-02 | -0.03%        |    32.03 |
| DQN_Average | 2014-02-24   | 2015-02-20 | 0.47%         |    31.39 |
| DQN_Average | 2018-03-09   | 2019-03-08 | 0.33%         |    30.81 |


### CVX Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | -0.02%        |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 0.33%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | -0.24%        |    20    |
| Base        | 2018-03-09   | 2019-03-08 | -0.07%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | -0.01%        |    30.87 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 0.33%         |    32.46 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -0.09%        |    30.64 |
| DQN_Average | 2018-03-09   | 2019-03-08 | -0.10%        |    30.19 |
| Base        | 2006-01-20   | 2007-01-19 | -0.02%        |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 0.33%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | -0.24%        |    20    |
| Base        | 2018-03-09   | 2019-03-08 | -0.07%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | -0.01%        |    30.32 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 0.33%         |    31.87 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -0.10%        |    29.64 |
| DQN_Average | 2018-03-09   | 2019-03-08 | -0.10%        |    30.16 |


### F Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | -0.37%        |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 1.49%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | 0.07%         |    20    |
| Base        | 2018-03-06   | 2019-03-05 | -0.38%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | -0.54%        |    27.93 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 6.37%         |    28.52 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -0.23%        |    28.71 |
| DQN_Average | 2018-03-06   | 2019-03-05 | -0.37%        |    28.84 |


### GOOG Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2007-09-07   | 2008-09-04 | 0.31%         |    20    |
| Base        | 2011-09-21   | 2012-09-18 | 0.15%         |    20    |
| Base        | 2015-10-08   | 2016-10-05 | -0.00%        |    20    |
| Base        | 2019-10-24   | 2020-10-21 | -0.01%        |    20    |
| DQN_Average | 2007-09-07   | 2008-09-04 | 0.26%         |    29.93 |
| DQN_Average | 2011-09-21   | 2012-09-18 | 0.00%         |    31.44 |
| DQN_Average | 2015-10-08   | 2016-10-05 | -0.01%        |    30.23 |
| DQN_Average | 2019-10-24   | 2020-10-21 | 0.00%         |    29.56 |


### M Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2010-06-18   | 2011-06-15 | -0.24%        |    20    |
| Base        | 2014-07-07   | 2015-07-02 | -0.54%        |    20    |
| Base        | 2018-07-20   | 2019-07-19 | -1.04%        |    20    |
| DQN_Average | 2010-06-18   | 2011-06-15 | 0.30%         |    29.01 |
| DQN_Average | 2014-07-07   | 2015-07-02 | -0.65%        |    30.46 |
| DQN_Average | 2018-07-20   | 2019-07-19 | -0.95%        |    30.26 |


### MSFT Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-19   | 2007-01-18 | 0.66%         |    20    |
| Base        | 2010-02-04   | 2011-02-01 | -0.43%        |    20    |
| Base        | 2014-02-21   | 2015-02-19 | 0.21%         |    20    |
| Base        | 2018-03-08   | 2019-03-07 | -0.43%        |    20    |
| DQN_Average | 2006-01-19   | 2007-01-18 | 1.75%         |    30.62 |
| DQN_Average | 2010-02-04   | 2011-02-01 | -0.81%        |    32.11 |
| DQN_Average | 2014-02-21   | 2015-02-19 | 1.41%         |    30.32 |
| DQN_Average | 2018-03-08   | 2019-03-07 | -0.38%        |    29.6  |


### NFLX Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | -0.76%        |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 8.22%         |    20    |
| Base        | 2014-02-24   | 2015-02-20 | 0.24%         |    20    |
| Base        | 2018-03-09   | 2019-03-08 | -0.25%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | -0.71%        |    27.88 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 10.38%        |    29.95 |
| DQN_Average | 2014-02-24   | 2015-02-20 | 0.29%         |    29.09 |
| DQN_Average | 2018-03-09   | 2019-03-08 | -0.43%        |    28.28 |


### NVDA Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | 6.25%         |    20    |
| Base        | 2010-02-05   | 2011-02-02 | 13.07%        |    20    |
| Base        | 2014-02-24   | 2015-02-20 | -0.95%        |    20    |
| Base        | 2018-03-09   | 2019-03-08 | -0.17%        |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | 8.33%         |    28.82 |
| DQN_Average | 2010-02-05   | 2011-02-02 | 17.29%        |    30.14 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -1.41%        |    28.97 |
| DQN_Average | 2018-03-09   | 2019-03-08 | -0.17%        |    27.91 |


### TGT Performance Results

| Episode     | Start Date   | End Date   | PnL% Change   |   Period |
|:------------|:-------------|:-----------|:--------------|---------:|
| Base        | 2006-01-20   | 2007-01-19 | 0.62%         |    20    |
| Base        | 2010-02-05   | 2011-02-02 | -0.73%        |    20    |
| Base        | 2014-02-24   | 2015-02-20 | -0.32%        |    20    |
| Base        | 2018-03-09   | 2019-03-08 | 0.17%         |    20    |
| DQN_Average | 2006-01-20   | 2007-01-19 | 0.29%         |    27.87 |
| DQN_Average | 2010-02-05   | 2011-02-02 | -1.10%        |    29.47 |
| DQN_Average | 2014-02-24   | 2015-02-20 | -0.01%        |    28.66 |
| DQN_Average | 2018-03-09   | 2019-03-08 | 0.43%         |    29.54 |