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


for s_alpha in $non_iid_alpha
do
    for s_ver in $v
    do 
        ./subsample_brun.sh "$s_alpha" "$s_ver" "$lr_init" "$aggregation" "$weight_decay" \
            "$loc_n_epoch" "$model_arch" "$momentum_factor" "$n_clients" "$start" "$repeat_gpu" \
            "$dataset" "$align_data" "$sample_ratio" "$round_to_add_fake_data" "$num_class" \
            "$load_opt" "$num_synthetic_images" "$num_images_train_synthetic" "$start_layer" \
            "$beta" "$random_shuffle" "$synthetic_epoch" "$synthetic_version" "$num_act_clients" "$loc" "$init_comm_round"
    done 
done 