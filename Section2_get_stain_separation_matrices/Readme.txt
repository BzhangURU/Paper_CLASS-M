The section 2 is for getting stain separation matrices for all slides such that adaptive stain separation can be performed in CLASS-M.
The stain separation matrices for 420 TCGA WSIs are already calculated and saved as
StepF_output_normalization_factors_based_on_maxH_maxE_H_E_exclude_portion_0.01_0.01_dataset_preprocessed.txt
If you are working on those 420 TCGA WSIs, you don't need to run section 2 again.

If you want to calculate new stain separation matrices, you need to follow the steps:

Step A: 
The step A code reads Whole Slide Images (WSIs) and outputs background color in R, G, B channels for each WSI. 
# Input:
# ORIGINAL_IMAGES_PATH is the path to the folder that saves all WSIs.
# excel_loc is the path to the Excel file that saves list of 420 TCGA WSIs.

# Output:
# OUTPUT_TXT_FILE_NAME saves txt file that contains background color in R, G, B channels. 
# The columns of output txt file are: 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6)'
# Each row saves each WSI's results.

# In this code, we set DIRECTLY_SET_255_AS_BACKGROUND_COLOR=True





Step B: 
# This step B code reads step A's output as input to get list of WSIs,
# and also reads all the foreground tiles in each WSI to get mean of RGB on OD space for each WSI. (absorption of RGB channel)
# You need to set process_image_start=1, process_image_end=420 if the dataset is originally from 420 TCGA WSIs, 
# OUTSIDE_POLYGON_TYPE='All_foreground_tiles' in Section 1's code and run it to get all the foreground tiles.
# You can set DOWNSAMPLE_RATE to a larger number to save less tiles and speed up process.

# Input: 
# INPUT_BACKGROUND_COLOR_TXT_PATH_NAME: output of step A
# INPUT_IMAGE_DIR: root directory that saves the tiles
# INPUT_IMAGE_LIST_PATH: directory of txt files that saves the list of tiles. Each txt saves one WSI's list of all foreground tiles.

# Output:
# OUTPUT_PATH: path of output
# OUTPUT_FILE_NAME: name of output file that saves the info of (output of step A) + mean of RGB on OD space(absorption of light on RGB channel)
# Columns are index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);
#                'OD_value_R(7);OD_value_G(8);OD_value_B(9)



Step C:
# In step C code, we read tiles to get covariance matrix of RGB on OD space.

# Input:
# INPUT_MEAN_OF_RGB_OD_VALUE_TXT_PATH_NAME: output of step B
# INPUT_IMAGE_DIR: root directory that saves the tiles
# INPUT_IMAGE_LIST_PATH: directory of txt files that saves the list of tiles. Each txt saves one WSI's list of all foreground tiles.

# Output:
# OUTPUT_FILE_NAME: output file that saves the output of step B + covariance matrix info
# Columns in output file are index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);
#                OD_value_R(7);OD_value_G(8);OD_value_B(9);cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18)





Step D:
# In step D, we read output from step C, and calculates eigen vectors and eigen values of covariance matrix for each WSI. 
# The eigen vector with largest eigen value forms axis X, 
# The eigen vector with second largest eigen value forms axis Y, 
# The eigen vector with smallest eigen value forms axis Z.
# The positive direction of X, Y and Z are carefully chosen. (Check code for details)
# The three axes are orthogonal to each other in theory.  
# Then each pixel has its own coordinates on X, Y, Z. The code saves pixel coordinates (distribution) on OD space for each WSI.

# Input:
# INPUT_RGB_ABSORPTION_COV_MAT_TXT_NAME: output of step C. 
# INPUT_IMAGE_DIR: root directory that saves the tiles
# INPUT_IMAGE_LIST_PATH: directory of txt files that saves the list of tiles. Each txt saves one WSI's list of all foreground tiles.

# Output:
# OUTPUT_SUBFOLDER: folder that saves the pixel coordinates (distribution) in X, Y, Z. Each WSI has its own distribution file.
# OUTPUT_FILE_NAME: output file that saves the output of step C + eigen vectors and eigen values of covariance matrices.
# OUTPUT_TRACK_PROCESS: file that records running progress. 
# Columns in output file are 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
#            'OD_value_R(7);OD_value_G(8);OD_value_B(9);'+\
#            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
#            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
#            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
#            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
#            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30)'






Step E:
# Find H, E vectors (angles on X-Y plane on OD space) for 420 TCGA ccRCC WSIs based on Macenko 2009 paper method. 

# Input:
# INPUT_EIGEN_VECTORS_VALUES_TXT_NAME: output of step D.
# INPUT_ROOT_PATH: another output from step D. folder that saves the pixel coordinates (distribution) in X, Y, Z. Each WSI has its own distribution file.

# Output:
# OUTPUT_SUBFOLDER: output folder that saves training records for each WSI.
# OUTPUT_TRACK_PROCESS: a summary of found H and E angles on X-Y plane on OD space for all WSIs. 
# OUTPUT_TXT_NAME: output file that saves the output of step D + H and E angles on X-Y plane on OD space + stain seperation matrix. 
# Columns in output file are 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
#            'OD_value_R(7);OD_value_G(8);OD_value_B(9);'+\
#            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
#            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
#            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
#            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
#            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30);H_E_angle_thresh(31);'+\
#            'H_exclude_portion(32);E_exclude_portion(33);H_angle_degree(34);E_angle_degree(35);'+\
#            'Matrix_RGBOD_value_to_HERes_r0Hc0R(36);r0Hc1G(37);r0Hc2B(38);r1Ec0R(39);r1Ec1G(40);r1Ec2B(41);r2RESc0R(42);r2RESc1G(43);r2RESc2B(44)'







Step F:
# This step F program calculates max H (top 1%, exclude values below 0.15), max E, then record this as normalization factor. 
# Then most H and E values would have range 0.0 - 1.0. Based on Macenko 2009 paper. 

# Input:
# INPUT_MATRIX_INFO_TXT: output of step E
# INPUT_IMAGE_DIR: root directory that saves the tiles
# INPUT_IMAGE_LIST_PATH: directory of txt files that saves the list of tiles. Each txt saves one WSI's list of all foreground tiles.

# Output:
# OUTPUT_TRACK_PROCESS: txt file that tracks the running process.
# OUTPUT_FILE_NAME: output file that saves the output of step E + Hematoxylin and Eosin normalization information
# Columns in output file are 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
#            'absorption_R(7);absorption_G(8);absorption_B(9);'+\
#            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
#            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
#            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
#            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
#            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30);H_E_angle_thresh(31);'+\
#            'H_exclude_portion(32);E_exclude_portion(33);H_angle_degree(34);E_angle_degree(35);'+\
#            'Matrix_RGBabsorption_to_HERes_r0Hc0R(36);r0Hc1G(37);r0Hc2B(38);r1Ec0R(39);r1Ec1G(40);r1Ec2B(41);r2RESc0R(42);r2RESc1G(43);r2RESc2B(44);'+\
#            'H_max(45);E_max(46);H_mean_normalized_by_max(47);E_mean_normalized_by_max(48);H_std_normalized_by_max(49);E_std_normalized_by_max(50)'

# The output can be directly read in our CLASS-M model for adaptive stain separation!!!!!!!!!!!!!!!!!!!!!!!







