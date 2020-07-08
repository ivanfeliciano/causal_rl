EPISODES=5000
EPISODES_NINE=10000
MOD=100
EXP=10

PERCENTAGE=low


# PARTITION=25
# ST=one_to_one

# # DETERMINISTA
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic

# ST=many_to_one
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic

# ST=one_to_many

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic


# PARTITION=75

# # DETERMINISTA
# ST=one_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic

# ST=many_to_one
# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic

# ST=one_to_many

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION --stochastic


PARTITION=75

# DETERMINISTA
ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition $PARTITION
