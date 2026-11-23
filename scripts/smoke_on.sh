set -e
C='--model Qwen/Qwen3-0.6B --mode on --epochs 2 --lr 3e-6 --clip 30 --n_sorry 16 --n_sr 8 --n_gsm 8 --eval_bs 16'
python rsd.py $C --frozen_teacher --out runs/smoke_on_frozen
python rsd.py $C --out runs/smoke_on_rec
