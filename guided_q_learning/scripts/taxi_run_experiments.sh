EPISODES=1000
MOD=5
EXP=100

time python run_taxi_experiments.py --plot --episodes $EPISODES --experiments $EXP --mod $MOD --partition 15 
time python run_taxi_experiments.py --plot --episodes $EPISODES --experiments $EXP --mod $MOD --stochastic --partition 15 --threshold -30
