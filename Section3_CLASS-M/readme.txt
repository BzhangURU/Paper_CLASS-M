# The CLASS-M_semi_sup_classification.py code is an example of our CLASS-M model for semi-supervised classification running on TCGA ccRCC dataset
# based on algorithms introduced in https://arxiv.org/abs/2312.06978

# Environment settings: Python 3.7.11, Pytorch 1.9.0, torchvision 0.10.0, and CUDA 10.2. The GPUs we used are NVIDIA TITAN RTX


# The hyper-parameters that can be fine-tuned are above "#######################..." line inside code. 
# The hyper-parameters in the code were already fine-tuned. 

# There are 4 program modes that can be set in "program_mode". 
# "normal_training" allows you to train CLASS-M model from scratch. 
# If the training process is unexpectedly stopped (like server shutdown...), 
# "resume_best_training" mode allows you to resume from best balanced validation epoch. 
# "resume_latest_training" mode allows you to resume from latest saved epoch. 
# We have "save_latest_epoch_frequency" to control model saving frequency.
# To just load already trained model and run validation/test again, choose "only_test" mode. 

# Input:
# INPUT_MATRIX_INFO_TXT: If LOAD_ADAPTIVE_HE_MATRIX is True, before running this code, you need to calculate stain separation matrix for each slide and load it to INPUT_MATRIX_PATH.
# If you don't want to calculate it, set LOAD_ADAPTIVE_HE_MATRIX to False, meanwhile the accuracy would drop a littile bit. 
# list_of_each_GP_txt_path: Each file inside path saves the tile list for each set (labeled/unlabeled train, val, test) and for each category (Normal tissue, Cancer, Necrosis)
# root_dir_original_images: folder that contains related training/validation/test tiles. 

# Output:
# save_models_folder: folder that will save trained models.
# save_results_folder: folder that will save experiment results on training, val, test.


# Other files/folders needed to run this program:
# Folders datasets_my_lib, pytorch_balanced_sampler should be in the same place as this python file.

# In root_dir_original_images, the relative path to each image is like: /Necrosis/IMG0281_TCGA-CJ-4923-01Z-00-DX1.ADD3D7FE-46D1-49AF-B6D4-1FD07B761EF1/
# 20230207_IMG0281_TCGA-CJ-4923-01Z-00-DX1.ADD3D7FE-46D1-49AF-B6D4-1FD07B761EF1_polygon_1_Necrosis_0.png


# In TCGA ccRCC dataset:
# Number of unlabeled training samples: 1,373,684
# Labeled training set: each class's sample number:
# [84578, 180471, 7932]
# Validation set: each class's sample number:
# [19638, 79382, 1301]
# Test set: each class's sample number:
# [15323, 62565, 6168]