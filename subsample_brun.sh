#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT

non_iid_alpha=${1?:Error: what is the non_iid_ratio}
v=${2?:Error: the version }
lr_init=${3?:Error: default lr is 0.05}
aggregation=${4?:Error: use scaffold or not}
weight_decay=${5?:Error: the weight decay for the bottleneck}
loc_n_epoch=${6?:Error: the number epoch per round}
model_arch=${7:-fc}
momentum_factor=${8:-0}
n_clients=${9:-18}
start=${10:-0}
repeat_gpu=${11:-8}
dataset=${12:-dsprint}
align_data=${13:-add_fake_samples}
sample_ratio=${14:-1}
round_to_add_fake_data=${15:-40}
num_class=${16:-10}
load_opt=${17:-train_latent}
num_synthetic_images=${18:-5000}
num_images_train_synthetic=${19:-4500}
start_layer=${20:-16}
beta=${21:-0.01}
random_shuffle=${22:-true}
synthetic_epoch=${23:-500}
synthetic_version=${24:-2}
num_act_clients=${25:-4}
loc=${26:-scratch}
init_comm_round=${27:-0}

num2=8
num3=16
num4=24
num5=32
lr_schedule=constant
use_wandb=false
 
end_commu=101

if [ "$align_data" == add_fake_diffusion_sync_local_generator ]; then
    fake_ratio=1
else 
    fake_ratio=0
fi 

if [ "$aggregation" == fed_pvr ]; then 
    start_layer=16
else    
    start_layer=0
fi 


if [ "$non_iid_alpha" == 0 ]; then
    partition_type=sort 
else
    partition_type=non_iid 
fi



for j in $(seq "$init_comm_round" 1 "$end_commu")
do
    python3 sample.py --local_n_epochs "$loc_n_epoch" \
        --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
        --communication_round "$j" --align_data "$align_data" \
        --lr_schedule "$lr_schedule" --version "$v" --loc "$loc" --arch "$model_arch" \
        --n_clients "$n_clients" --aggregation "$aggregation" --num_class "$num_class" \
        --partition_type "$partition_type" --data "$dataset" --num_synthetic_images "$num_synthetic_images" \
        --num_images_train_synthetic "$num_images_train_synthetic" --synthetic_version "$synthetic_version" \
        --start_layer "$start_layer" --random_shuffle "$random_shuffle" --synthetic_epoch "$synthetic_epoch" \
        --sample_ratio "$sample_ratio" --beta "$beta"

    for i in $(seq "$start" 1 "$((num_act_clients-1+start))")
    do
        if [ "$i" -lt "$num2" ]; then
            gpu_index="$i"
        elif [ "$i" -ge "$num2" ] && [ "$i" -lt "$num3" ]; then 
            gpu_index="$((i-repeat_gpu))"
        elif [ "$i" -ge "$num3" ] && [ "$i" -lt "$num4" ]; then 
            gpu_index="$((i-num3))"
        elif [ "$i" -ge "$num4" ] && [ "$i" -lt "$num5" ]; then 
            gpu_index="$((i-num4))"
        else
            gpu_index="$((i-num5))"
        fi
        i="$((i-start))"
        echo "$gpu_index"
        export CUDA_VISIBLE_DEVICES="$gpu_index"
        
        python3 create_train.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
            --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
            --communication_round "$j" --loc "$loc" \
            --lr_schedule "$lr_schedule" --version "$v" --use_wandb "$use_wandb" --arch "$model_arch" \
            --partition_type "$partition_type" \
            --data "$dataset" --align_data "$align_data" --num_class "$num_class" \
            --num_synthetic_images "$num_synthetic_images" --synthetic_epoch "$synthetic_epoch" \
            --num_images_train_synthetic "$num_images_train_synthetic" --synthetic_version "$synthetic_version" \
            --start_layer "$start_layer" --beta "$beta" --random_shuffle "$random_shuffle" \
            --sample_ratio "$sample_ratio" --n_clients "$n_clients" --aggregation "$aggregation" --momentum_factor "$momentum_factor" &
    done
    wait 

    echo "Done training all the clients"
    for i in $(seq 0 1 "$((num2-1))")
    do
        export CUDA_VISIBLE_DEVICES="$i"
        if [ "$i" == 0 ]; then 
            worker_for_occupy_gpu=false
        else
            worker_for_occupy_gpu=true 
        fi 
        python3 communicate.py --use_local_id "$i" --local_n_epochs "$loc_n_epoch" \
            --lr "$lr_init" --non_iid_alpha "$non_iid_alpha" --weight_decay "$weight_decay" \
            --communication_round "$j" --loc "$loc" \
            --lr_schedule "$lr_schedule" --version "$v" --use_wandb "$use_wandb" --arch "$model_arch" \
            --partition_type "$partition_type" --align_data "$align_data" --num_class "$num_class" \
            --data "$dataset" --worker_for_occupy_gpu "$worker_for_occupy_gpu" \
            --num_synthetic_images "$num_synthetic_images" --synthetic_epoch "$synthetic_epoch" \
            --num_images_train_synthetic "$num_images_train_synthetic" \
            --synthetic_version "$synthetic_version" --sample_ratio "$sample_ratio" \
            --start_layer "$start_layer" --beta "$beta" --random_shuffle "$random_shuffle" \
            --n_clients "$n_clients" --aggregation "$aggregation" --momentum_factor "$momentum_factor" &
    done
    wait 
    echo "Done communicating"
done