EPISODES=500
MOD=20
EXP=20

ST=one_to_one

PERCENTAGE=low
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=medium
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=high
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d


ST=many_to_one

PERCENTAGE=low
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=medium
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=high
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d

ST=one_to_many

PERCENTAGE=low
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=medium
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d
PERCENTAGE=high
time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP --pmod $PERCENTAGE -d



# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP


# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP


# ST=many_to_one

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP


# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP

# ST=one_to_many

# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --experiments $EXP


# time python run_light_env_experiments.py --num 5 --structure $ST --episodes $EPISODES --mod $MOD --stochastic
# time python run_light_env_experiments.py --num 7 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP
# time python run_light_env_experiments.py --num 9 --structure $ST --episodes $EPISODES --mod $MOD --stochastic --experiments $EXP
