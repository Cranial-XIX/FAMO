mkdir -p ./save
mkdir -p ./trainlogs

method=famo
seed=42
gamma=0.001

python trainer.py --method=$method --seed=$seed --gamma=$gamma --scale-y=True > trainlogs/famo-gamma$gamma-$seed.log 2>&1 &
