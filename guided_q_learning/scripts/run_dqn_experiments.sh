EPISODES=200
MOD=1
EXP=10
PARTITION=75
DIR_NAME=/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches/dqn


ST=one_to_one
time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic

ST=one_to_many

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic

ST=many_to_one

time python experiments_dqn.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic
time python experiments_dqn.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --partition $PARTITION --basedir $DIR_NAME --draw --stochastic

