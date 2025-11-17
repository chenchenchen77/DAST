# model_id=/data/chenss/UltraGist/data/outputs/ultragist-llama2-7b-chat-ft/checkpoint-1000
model_id=/data/chenss/ultragist_data/outputs/ultragist-llama2-7b-chat-ft/checkpoint-1000
# you can also evaluate your models
model_id=/data/chenss/llama2-ultragist
model_id=/data/chenss/llama2-ultragist/base-llama2
# model_id=/data/chenss/llama2-ultragist/base_llalam3/checkpoint-3509
model_id=/data/chenss/llama2-ultragist/query-llama2/checkpoint-21914
export CUDA_VISIBLE_DEVICES="1"
# model_id=data/outputs/ultragist-llama2-7b-chat-ft/checkpoint-xxxx
device=1



id=29520
########## Topic Retrieval ##########

# for ratio in 4 8 16 24
# for ratio in 4
# do
# # the default window 1024 cannot be evenly divided by 24, so we change it to 960
# if [[ $ratio == "24" ]]; then
#     torchrun --nproc_per_node $device -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10 --ultragist_window 960 --ultragist_stride 960 --ultragist_ratio_mix adapt-1024
# else
#     torchrun --master_port 29502 --nproc_per_node $device -m main.eval_topic --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --num_topic 1 2 3 10 --ultragist_ratio_mix adapt-1024
# fi
# done

########## MSC ##########
# for ratio in 4 8 16
# do
# # the default window 1024 cannot be evenly divided by 24, so we change it to 960
# if [[ $ratio == "24" ]]; then
#     torchrun --nproc_per_node $device -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no --ultragist_window 960 --ultragist_stride 960 #--ultragist_ratio_mix adapt-1024
# else
#     torchrun --nproc_per_node $device -m main.eval_msc --ultragist_ratio $ratio --model_name_or_path $model_id --enable_ultragist --chat_template no #--ultragist_ratio_mix adapt-1024
# fi
# done

# ########### Long-Context Tasks ##########
# TODO
# torchrun --nproc_per_node $device --master_port $id -m main.eval_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 0 2 4 8 16 32 --ultragist_ratio_mix adapt-1024 --ultragist_attn full-coverage
torchrun --nproc_per_node $device --master_port $id -m main.eval_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 0 2 4 8 16 32 --ultragist_ratio_mix adapt-1024 #--chat_template llama-3 #--ultragist_attn full-coverage 
# torchrun --nproc_per_node $device --master_port $id -m main.query_longbench --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 0 2 4 8 16 32 --ultragist_ratio_mix adapt-1024 #--chat_template llama-3 #--ultragist_attn full-coverage 

# ########### Needle-In-A-Haystack ##########
# torchrun --nproc_per_node $device -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 32000 --ultragist_ratio 2 4 8 --ultragist_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2

# # by default, we evaluate with ROUGE-L (R), you can specify OPENAI_API_KEY to use gpt-3.5 as evaluator
# # OPENAI_API_KEY="<you_api_key>" torchrun --nproc_per_node 8 -m main.eval_needle --model_name_or_path $model_id --enable_ultragist --max_length 32000 --ultragist_ratio 2 4 8 --beacon_ratio_mix adapt-1024 --rope_method dynamic --rope_factor 2 --gpt_eval

# ########## ShareGPT ##########
# for turn in 1 2 3
# do
# torchrun --nproc_per_node $device -m main.eval_multiturn --model_name_or_path $model_id --enable_ultragist --ultragist_ratio 8 --ultragist_window 512 --ultragist_stride 512 --num_turn $turn --ultragist_ratio_mix adapt-1024
# done


#有些数据集应该是小于最大position不压缩了
#1024 -> 不压缩
#试试1024直接给1024个压缩token

# future token
# 加个判断，首先如何存在max_length = 4096 -> 先简单判断
# 那么直接选即可， 否则再执行接下来的