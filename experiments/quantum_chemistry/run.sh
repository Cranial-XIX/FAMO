mkdir -p ./save
mkdir -p ./trainlogs

method=famo
seed=42
gamma=0.01

#python trainer.py --method=$method --seed=$seed --gamma=$gamma > trainlogs/famo-gamma$gamma-$seed.log 2>&1 &
python trainer.py --method=$method --seed=$seed --gamma=$gamma
