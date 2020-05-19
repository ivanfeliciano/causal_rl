EPISODES=200
MOD=10
EXPERIMENTS=10
ST=one_to_one

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXPERIMENTS
