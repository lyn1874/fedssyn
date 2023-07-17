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
fake_ratio=${14:-1}
round_to_add_fake_data=${15:-40}
num_class=${16:-10}
load_opt=${17:-train_latent}
num_synthetic_images=${18:-5000}
num_images_train_synthetic=${19:-4500}
syn_epoch=${20:-500}
syn_version=${21:-2}
s_layer=${22:-0}
loc=${23:-scratch}
beta=${24:-0.1}

for s_epoch in $syn_epoch
do
    for s_alpha in $non_iid_alpha
    do
        for s_im_tr in $num_images_train_synthetic
        do
            for s_im in $num_synthetic_images
            do 

                ./brun.sh "$s_alpha" "$v" "$lr_init" "$aggregation" "$weight_decay" "$loc_n_epoch" "$model_arch" \
                    "$momentum_factor" "$n_clients" "$start" "$repeat_gpu" "$dataset" "$align_data" "$fake_ratio" \
                    "$round_to_add_fake_data" "$num_class" "$load_opt" "$s_im" "$s_im_tr" "$s_layer" "$beta" false "$s_epoch" \
                    "$syn_version" "$loc"
            done 
        done 
    done 
done 