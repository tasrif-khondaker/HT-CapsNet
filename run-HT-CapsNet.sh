#!/bin/bash

#SBATCH --job-name=HT-CapsNet

# Print debugging information
echo "###############################################################"
echo "||                    Information - Start                    ||"
echo "###############################################################"
echo "Date                          : $(date)"
echo "Hostname                      : $(hostname -s)"
echo "Working Directory             : $(pwd)"
echo "Number of nodes used          : $SLURM_NNODES"
echo "Number of threads             : $SLURM_CPUS_PER_TASK"
echo "Name of nodes used            : $SLURM_JOB_NODELIST"
echo "Gpu devices                   : $CUDA_VISIBLE_DEVICES"
echo "###############################################################"

###################################################################
# >>>>>>>>>>>>>>>> Define hyperparameters <<<<<<<<<<<<<<<<<<<<<<<<<
###################################################################
datasets=("F_MNIST" "CIFAR10" "CIFAR100" "S_Cars" "CU_Birds" "M_Tree_L4" "M_Tree_L3" "M_Tree_L2" "M_Tree_L1") # Choices: ("MNIST" "E_MNIST" "F_MNIST" "CIFAR10" "CIFAR100" "S_Cars" "CU_Birds" "M_Tree")
data_augments=("MixUp") # Choices: ("MixUp" "H_MixUp" "H_CutMix" "H_MixupAndCutMix")
modes=("HTR_CapsNet") # Choices: ("BUH_CapsNet" "HD_CapsNet" "ML_CapsNet" "HDR_CapsNet" "HD_CapsNet_Eff" "HTR_CapsNet" "HD_CapsNet_EM")

SCaps_dims=("64" "32" "16") # Choices: ("16" "32" "64" "128" "256")
SCaps_dim_modes=("decrease") # Choices: ("same" "increase" "decrease")

backbones=("EfficientNetB7") # Choices: ("custom" "MobileNetV2" "EfficientNetB7" "InceptionV3" "ResNet50V2" "VGG16" "VGG19" "Xception" "DenseNet121")
backbone_weights=("imagenet") # Choices: ("imagenet" "None")
Routing_Nos=("2" "3" "4" "5")

epochs=200
batch_size=32
gpu_id=0
###################################################################
###################################################################

# Main job submission loop
for dataset in "${datasets[@]}"; do
    for data_augment in "${data_augments[@]}"; do
        for backbone in "${backbones[@]}"; do
            for backbone_weight in "${backbone_weights[@]}"; do
                for mode in "${modes[@]}"; do
                    for SCaps_dim in "${SCaps_dims[@]}"; do
                        for SCaps_dim_mode in "${SCaps_dim_modes[@]}"; do
                            for Routing_No in "${Routing_Nos[@]}"; do
                                echo "###############################################################"
                                echo " Current Job Details:                                       "
                                echo "                - Dataset:  $dataset                        "
                                echo "                - Data Augmentation:  $data_augment         "
                                echo "                - Mode:  $mode                              "
                                echo "                - Routing_No:  $Routing_No                  "
                                echo "                - SCaps_dim:  $SCaps_dim                    "
                                echo "                - SCaps_dim_mode:  $SCaps_dim_mode          "
                                echo "                - Backbone:  $backbone                      "
                                echo "                - Backbone Weight:  $backbone_weight        "
                                echo "###############################################################" 
                                # Run the notebook
                                papermill main.ipynb output-"$dataset"-"$data_augment"-"$backbone"-"$backbone_weight"-"$mode"-"$SCaps_dim-$SCaps_dim_mode"-R-"$Routing_No".ipynb \
                                    -p args_dict '{
                                        "dataset": "'$dataset'", 
                                        "mode": "'$mode'", 
                                        "SCaps_dim": '$SCaps_dim',
                                        "SCaps_dim_mode": "'$SCaps_dim_mode'",
                                        "Routing_N": "'$Routing_No'",
                                        "data_aug": "'$data_augment'", 
                                        "backbone_net": "'$backbone'", 
                                        "backbone_net_weights": "'$backbone_weight'", 
                                        "epochs": '$epochs',
                                        "gpus": "'$gpu_id'",
                                        "batch_size": '$batch_size',
                                        "dir_uid": "S_dim_'$SCaps_dim'_'$SCaps_dim_mode'_R_'$Routing_No'",
                                        "data_path": "/Dataset/Marine_tree" # Dataset path on 
                                    }'
                            done
                        done
                    done
                done
            done
        done
    done
done