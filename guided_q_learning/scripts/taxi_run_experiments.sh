EPISODES=1000
MOD=10
EXP=10

time python run_taxi_experiments.py --plot --episodes $EPISODES --experiments $EXP --mod $MOD 
time python run_taxi_experiments.py --plot --episodes $EPISODES --experiments $EXP --mod $MOD --stochastic

