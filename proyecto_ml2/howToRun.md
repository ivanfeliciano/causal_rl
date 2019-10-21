# Action selection based on deterministic Causal Model (Taxi Domain)

We develope a Q-learning version where action selection is based on
a simple set of equation that represents a causal model of state, goal and action variables for the classic taxi problem.

## Usage

```python
python app.py
```

### Optional parameters

```bash
"--stochastic", help="change to simple stochastic environments (0.7 prob of do the choosen action)", action="store_true"
"--episodes", type=int, default=1000, help="# of episodes per experiment"
"--experiments", type=int, default=5, help="# of experiments"
"--stat_test", action="store_true", help="Compute the Mann-Whitney rank test to check if some algorithm reach the goal faster. Need > 20 experiments"
"-v", "--verbose", action="store_true", help="Verbosity activated"
```
## Output

The output of the program are several plots comparing the performance of each version of the algorithm.

![Comparison Deterministic Environment][comparisonDeterministic.jpg]
![Comparison Stochastic Environment][comparisonStochastic.jpg]


If significance test is required also the plots for the episode where the goal reward is reached are available.

![Goal Reward Deterministic][./goal_reward_comparisonDeterministic.jpg]
![Goal Reward Stochastic][./goal_reward_comparisonStochastic.jpg]


