EPISODES=5000
EPISODES_NINE=10000
MOD=100
EXP=10
DIR_NAME=/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches/discrete/masterswitch
ST=masterswitch
PERCENTAGE=low

# DETERMINISTA
# DELTA=1
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw

DELTA=25
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw

# DELTA=50
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw

# DELTA=100
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $DELTA --basedir $DIR_NAME --draw
