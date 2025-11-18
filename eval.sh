model_id=llama2/checkpoint-21914
export CUDA_VISIBLE_DEVICES="0"
device=1
id=29520
torchrun --nproc_per_node $device --master_port $id -m main.eval_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 0 2 4 8 16 32 --ultragist_ratio_mix adapt-1024
