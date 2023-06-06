mkdir -p ./save
mkdir -p ./trainlogs

method=famo
seed=42
gamma=0.0 # for the toy example, we need to use a very small gamma as the objective is not changing

python trainer.py --method=$method --seed=$seed --gamma=$gamma > trainlogs/famo-gamma$gamma-$seed.log 2>&1 &
