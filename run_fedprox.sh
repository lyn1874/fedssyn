#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT
g=${1?:-8}
data=${2?:-cifar100}
align=${3:-add_fake_diffusion_sync_local_generator_without_shuffle}
v_g=${4:-30}
n_devices=${5:-10}
loc=${6:-nobackup}
beta=${7:-0.01}

beta=0.01
if [ "$data" == cifar10 ]; then 
    lr=0.05
else   
    lr=0.1 
fi 


if [ "$n_devices" == 10 ]; then 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./run_diff_gen.sh 0.01 "$s_v" "$lr" fed_prox 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 0 0 10 train_im 0 0 0 0 0 "$loc" "$beta" 
            ./run_diff_gen.sh 0.1 "$s_v" "$lr" fed_prox 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 0 0 10 train_im 0 0 0 0 0 "$loc" "$beta" 
        done 

    elif [ "$align" == add_fake_diffusion_sync_local_generator ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g
        do
            ./run_diff_gen.sh 0.01 "$s_v" 0.1 fed_prox 0 10 VGG_11 0 10 0 8 "$data" "$align" 1 101 10 train_im 5000 3375 450 2 0 "$loc" "$beta" 
            ./run_diff_gen.sh 0.1 "$s_v" 0.1 fed_prox 0 10 VGG_11 0 10 0 8 "$data" "$align" 1 101 10 train_im 5000 3375 450 2 0 "$loc" "$beta" 
        done 
    fi     

elif [ "$n_devices" == 4 ]; then 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr" fed_prox 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 0 10 train_im 0 0 0 "$beta" false 0 0 4 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr" fed_prox 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 0 10 train_im 0 0 0 "$beta" false 0 0 4 "$loc" 0 
        done 

    elif [ "$align" == add_fake_diffusion_sync_local_generator ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" 0.1 fed_prox 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 101 10 train_im 5000 3375 0 "$beta" false 450 2 4 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" 0.1 fed_prox 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 101 10 train_im 5000 3375 0 "$beta" false 450 2 4 "$loc" 0 
        done 
    fi     
elif [ "$n_devices" == 40 ]; then 
    if [ "$data" == cifar10 ]; then 
        loc_epoch=30
        s_epoch=450
    elif [ "$data" == cifar100 ]; then 
        loc_epoch=40 
        s_epoch=499
    fi 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr" fed_prox 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                none 0.2 0 10 train_im 0 0 0 "$beta" false 0 0 8 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr" fed_prox 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                none 0.2 0 10 train_im 0 0 0 "$beta" false 0 0 8 "$loc" 0 
        done 
    elif [ "$align" == add_fake_diffusion_sync_local_generator ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" 0.1 fed_prox 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                "$align" 0.2 101 10 train_im 1125 845 0 "$beta" false "$s_epoch" 2 8 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" 0.1 fed_prox 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                "$align" 0.2 101 10 train_im 1125 845 0 "$beta" false "$s_epoch" 2 8 "$loc" 0 
        done 
    fi     

elif [ "$n_devices" == 100 ]; then 
    if [ "$data" == cifar10 ]; then 
        loc_epoch=30
        s_epoch=499
        num_im=500
    elif [ "$data" == cifar100 ]; then 
        loc_epoch=40 
        s_epoch=499
        num_im=1000
    fi 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr" fed_prox 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                none 0.1 0 10 train_im 0 0 0 "$beta" false 0 0 10 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr" fed_prox 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                none 0.1 0 10 train_im 0 0 0 "$beta" false 0 0 10 "$loc" 0 
        done 
    elif [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ] || [ "$align" == add_fake_diffusion_sync_local_generator ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr" fed_prox 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                "$align" 0.1 101 10 train_im "$num_im" 385 0 "$beta" false "$s_epoch" 3 10 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" 0.1 fed_prox 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                "$align" 0.1 101 10 train_im "$num_im" 385 0 "$beta" false "$s_epoch" 3 10 "$loc" 0 

        done 
    fi     

fi 

