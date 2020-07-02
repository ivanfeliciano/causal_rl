EPISODES=10000
EPISODES_NINE=20000
MOD=100
EXP=10

PERCENTAGE=low

ST=one_to_one

echo $PERCENTAGE 
# DETERMINISTA
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

# ESTOCASTICO

ST=one_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic

ST=many_to_one

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic


ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic

PERCENTAGE=medium

echo $PERCENTAGE
ST=one_to_one

# DETERMINISTA
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

# # ESTOCASTICO

# ST=one_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic

# ST=many_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic


# ST=one_to_many

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic


PERCENTAGE=high

echo $PERCENTAGE

ST=one_to_one

# DETERMINISTA
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=many_to_one
time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

ST=one_to_many

time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50
time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50

# # ESTOCASTICO

# ST=one_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic

# ST=many_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic


# ST=one_to_many

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES_NINE --mod $MOD --experiments $EXP --pmod $PERCENTAGE --partition 50 --stochastic

