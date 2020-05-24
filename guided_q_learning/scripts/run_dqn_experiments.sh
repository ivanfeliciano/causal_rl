EPISODES=1000
MOD=50
EXP=100
ST=one_to_one

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw


time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw


ST=many_to_one

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw


time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw

ST=one_to_many

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --draw


time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --draw
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP --draw
