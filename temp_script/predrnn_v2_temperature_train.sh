cd ..
.venv/bin/python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name temperature \
    --train_data_paths data/temperature \
    --valid_data_paths data/temperature \
    --save_dir checkpoints/temperature_predrnn_v2 \
    --gen_frm_dir results/temperature_predrnn_v2 \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 256 \
    --img_channel 1 \
    --input_length 3 \
    --total_length 5 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 2500 \
    --r_sampling_step_2 5000 \
    --r_exp_alpha 1000 \
    --lr 0.0001 \
    --batch_size 6 \
    --max_iterations 10000 \
    --display_interval 50 \
    --test_interval 1000 \
    --snapshot_interval 1000 \
#    --pretrained_model ./checkpoints/temperature_predrnn_v2/temperature_model.ckpt

