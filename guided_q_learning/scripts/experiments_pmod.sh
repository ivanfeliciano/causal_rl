EPISODES=10000
EPISODES_NINE=20000
MOD=1
EXP=10
DIR_NAME=/home/ivan/Documentos/causal_rl/guided_q_learning/results_thesis/light_switches/discrete/experiment_pmod


PERCENTAGE=low

# DETERMINISTA
ST=one_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

# ESTOCASTICO

ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw

ST=many_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw


ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw


PERCENTAGE=medium

# DETERMINISTA
ST=one_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

# ESTOCASTICO

ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw

ST=many_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw


ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw


PERCENTAGE=high

# DETERMINISTA
ST=one_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --basedir $DIR_NAME --draw

# ESTOCASTICO

ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw

ST=many_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw


ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic --basedir $DIR_NAME --draw
