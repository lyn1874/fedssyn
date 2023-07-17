#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT
g=${1?:-8}
data=${2?:-cifar100}
v_g=${3:-30}
n_devices=${4:-10}
loc=${5:-nobackup}
agg=${6:-fed_avg}
align=${7:-add_fake_diffusion_sync_local_generator}


if [ "$agg" == fed_avg ]; then 
    if [ "$data" == cifar10 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.1 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.05 #0.01 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1

            fi 
        else
            lr_alpha_l=0.1 
            lr_alpha_h=0.1 
        fi 

    elif [ "$data" == cifar100 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.1 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.05 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            fi 
        else
            lr_alpha_l=0.1 
            lr_alpha_h=0.1 
        fi 
    fi 
elif [ "$agg" == scaffold ]; then 
    if [ "$data" == cifar10 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.05 #0.01 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.05 #0.01 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1

            fi 
        else
            if [ "$n_devices" == 4 ]; then 
                lr_alpha_l=0.05 
                lr_alpha_h=0.05
            else
                lr_alpha_l=0.1 
                lr_alpha_h=0.1
            fi 
        fi 
    elif [ "$data" == cifar100 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.1 
                lr_alpha_h=0.1
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.05 
                lr_alpha_h=0.01
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            fi 
        else
            lr_alpha_l=0.1 
            lr_alpha_h=0.1 
        fi 
    fi 

elif [ "$agg" == fed_pvr ]; then 
    if [ "$data" == cifar10 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.1 #0.05 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.05 
                lr_alpha_h=0.01
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1

            fi 
        else
            lr_alpha_l=0.1 
            lr_alpha_h=0.1
        fi 
    elif [ "$data" == cifar100 ]; then 
        if [ "$align" == none ] || [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then
            if [ "$n_devices" == 10 ]; then 
                lr_alpha_l=0.1 
                lr_alpha_h=0.05
            elif [ "$n_devices" == 4 ]; then   
                lr_alpha_l=0.01 
                lr_alpha_h=0.01
            elif [ "$n_devices" == 40 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1
            elif [ "$n_devices" == 100 ]; then 
                lr_alpha_l=0.1
                lr_alpha_h=0.1

            fi 
        else
            lr_alpha_l=0.1 
            lr_alpha_h=0.1 
        fi 
    fi 
fi 

beta=0

if [ "$n_devices" == 10 ]; then 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./run_diff_gen.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 0 0 10 train_im 0 0 0 0 0 "$loc" "$beta" 
            ./run_diff_gen.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 0 0 10 train_im 0 0 0 0 0 "$loc" "$beta" 
        done 

    elif [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g
        do
            ./run_diff_gen.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 1 101 10 train_im 5000 3375 450 2 0 "$loc" "$beta" 
            ./run_diff_gen.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 10 VGG_11 0 10 0 "$g" "$data" "$align" 1 101 10 train_im 5000 3375 450 2 0 "$loc" "$beta" 
        done 
    fi     

elif [ "$n_devices" == 4 ]; then 
    if [ "$align" == none ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 0 10 train_im 0 0 0 "$beta" false 0 0 4 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 0 10 train_im 0 0 0 "$beta" false 0 0 4 "$loc" 0 
        done 

    elif [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 10 VGG_11 0 10 "$g" 8 "$data" \
                "$align" 0.4 101 10 train_im 5000 3375 0 "$beta" false 450 2 4 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 10 VGG_11 0 10 "$g" 8 "$data" \
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
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                none 0.2 0 10 train_im 0 0 0 "$beta" false 0 0 8 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                none 0.2 0 10 train_im 0 0 0 "$beta" false 0 0 8 "$loc" 0 
        done 
    elif [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
                "$align" 0.2 101 10 train_im 1125 845 0 "$beta" false "$s_epoch" 2 8 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 "$loc_epoch" VGG_11 0 40 "$g" 8 "$data" \
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
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                none 0.1 0 10 train_im 0 0 0 "$beta" false 0 0 10 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                none 0.1 0 10 train_im 0 0 0 "$beta" false 0 0 10 "$loc" 0 
        done 
    elif [ "$align" == add_fake_diffusion_sync_local_generator_without_shuffle ] || [ "$align" == add_fake_diffusion_sync_local_generator ]; then 
        for s_v in $v_g 
        do
            ./multi_version_subsample_brun.sh 0.01 "$s_v" "$lr_alpha_l" "$agg" 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                "$align" 0.1 101 10 train_im "$num_im" 385 0 "$beta" false "$s_epoch" 3 10 "$loc" 0 
            ./multi_version_subsample_brun.sh 0.1 "$s_v" "$lr_alpha_h" "$agg" 0 "$loc_epoch" VGG_11 0 100 "$g" 8 "$data" \
                "$align" 0.1 101 10 train_im "$num_im" 385 0 "$beta" false "$s_epoch" 3 10 "$loc" 0 
        done 
    fi     

fi 

