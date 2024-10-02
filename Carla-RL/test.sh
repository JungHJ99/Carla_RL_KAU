python train_sac.py \
    --model-name logs/sac_model_350000_steps \
    --map Town01 \
    --width 160 \
    --height 80 \
    --repeat-action 4 \
    --start-location fixed \
    --sensor semantic \
    --episode-length 2000 \
    --load True