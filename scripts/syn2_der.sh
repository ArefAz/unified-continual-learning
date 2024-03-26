CUDA_VISIBLE_DEVICES=0 WANDB_MODE="dryrun" /home/aref/projects/venv/bin/python3.9 -m main.main --dataset=syn-2  --model=udil --lr=5e-5 \
    --n-epochs=150 \
    --batch-size=64  --backbone=densenet \
    --loss=ce --ignore-other-metrics 1 \
    --checkpoint --num-workers=8 \
    --nowand \
    --opt=adam --seed=0 \
    --wandb-name=cl-er-10ep-1000 \
    --buffer-size=1000 --buffer-batch-size=64 \
    --discriminator=mnistmlp --new-pkl " " \
    --disc-num-layers=4 --disc-k=1 --disc-lr=1e-5 \
    --task-weight-k=1 --task-weight-lr=2e-3 --loop-k=1 --epoch-scaling=const --C=5 \
    --visualize --checkpoint --encoder-lambda=0.1 --encoder-mu=10 \
    --loss-form=sum --supcon-lambda=0.1 --supcon-temperature=0.07 --kd-threshold=2 \
    --supcon-sim=l2 --supcon-first-domain
    # --validation 
