CUDA_VISIBLE_DEVICES=0 WANDB_MODE="dryrun" /home/aref/projects/venv/bin/python3.9 -m main.main --dataset=syn-2 --model=udil --lr=1e-5 --n-epochs=100 \
    --batch-size=64 --buffer-size=1000 --buffer-batch-size=64 --backbone=mislnet \
    --discriminator=mnistmlp --disc-num-layers=4 --loss=ce --disc-k=1 --disc-lr=1e-5 \
    --task-weight-k=1 --task-weight-lr=2e-3 --loop-k=1 --epoch-scaling=const --C=5 \
    --checkpoint --encoder-lambda=2 --num-workers=8 \
    --encoder-mu=50 --opt=adam --loss-form=sum --seed=1208 \
    --supcon-lambda=0.01 --supcon-temperature=0.07 --kd-threshold=2 \
    --supcon-sim=l2 --supcon-first-domain --wandb-name=PermMNIST-UDIL --disc-hiddim=800