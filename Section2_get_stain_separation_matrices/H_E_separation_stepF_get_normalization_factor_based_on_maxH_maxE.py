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

#around 4 WSI/minute
import os
from PIL import Image
from numpy import linalg as LA
import numpy as np
import torch
from datetime import datetime

TOTAL_NUM_WSIS=420
process_image_start=1#min 1
process_image_end=420#max TOTAL_NUM_WSIS, included, for TCGA
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with all lines. 
DOWNSAMPLE_RATE=10#downsample each tile
max_image_num_thresh=-1#-1(all), 5
H_PORTION_LOW_THRESH=0.15#below this value, we think it is more influenced by noise. we don't use in calculation.
E_PORTION_LOW_THRESH=0.15#below this value, we think it is more influenced by noise. we don't use in calculation.
GET_TOP_PERCENT=1.0#Ex: 6.0 means we get the H value that is top 6.0% in H portion among samples above H_PORTION_LOW_THRESH
ADDITIONAL_NORM_FACTOR=2.0
PREPROCESS_DATASET=True
H_exclude_portion=0.01
E_exclude_portion=0.01
if PREPROCESS_DATASET:
    #we may delete some pixels less than LOW_THRESH from original point
    preprocess_str='_dataset_preprocessed'
else:
    preprocess_str=''
str_show_settings='H_E_exclude_portion_'+'{:.2f}'.format(H_exclude_portion)+'_'+'{:.2f}'.format(E_exclude_portion)

OUTPUT_FILE_NAME='StepF_output_normalization_factors_based_on_maxH_maxE_'+str_show_settings+preprocess_str+'.txt'
OUTPUT_TRACK_PROCESS='StepF_output_track_running_process_'+str_show_settings+preprocess_str+'.txt'

INPUT_MATRIX_INFO_TXT='StepE_output_H_E_angles_'+str_show_settings+preprocess_str+'.txt'
INPUT_IMAGE_DIR='/your_own_path/tiles_20230301_1_20X/'
INPUT_IMAGE_LIST_PATH='/your_own_path/tiles_20230301_1_20X/All_foreground_tiles/'

ADD_STEP_TO_STRING=True

def get_normalization_factor_maxH_maxE_from_tiles_in_one_WSI(INPUT_IMAGE_DIR, input_image_list_txt, background_RGB_matrix, matrix_RGB2HE, f_track):
    f_input_image_list_txt = open(input_image_list_txt, 'r')
    one_image_relative_path = f_input_image_list_txt.readline()
    one_image_relative_path=one_image_relative_path.replace('\n','')
    background_RGB=np.array(background_RGB_matrix)#list
    RGB_absorption_to_H=np.array(matrix_RGB2HE[0])#numpy array
    RGB_absorption_to_E=np.array(matrix_RGB2HE[1])
    count=0

    while len(one_image_relative_path)>=3:
        count+=1
        if ADD_STEP_TO_STRING:
            one_image_relative_path=one_image_relative_path.replace('All_foreground_tiles/','All_foreground_tiles_sample_step_800/')
        cur_img_dir=INPUT_IMAGE_DIR+one_image_relative_path
        image_original = np.asarray(Image.open(os.path.join(cur_img_dir)))#(height, width, channel)
        image_original = image_original.astype('float32')
        image_downsampled=image_original[::DOWNSAMPLE_RATE,::DOWNSAMPLE_RATE,:]
        image_reshaped=np.reshape(image_downsampled,(-1,3))

        image_ones=np.ones(image_reshaped.shape,dtype=np.float32)
        image_reshaped = np.maximum(image_reshaped,image_ones)#change all pixel intensity =0 to =1
        
        RGB_absorption=np.log10((image_ones/image_reshaped)*background_RGB)
        gray_image_H=np.dot(RGB_absorption,RGB_absorption_to_H)
        bool_H=gray_image_H>=H_PORTION_LOW_THRESH
        gray_image_H_for_stat=gray_image_H[bool_H]
        if count==1:
            print('Check first gray_image_H_for_stat shape: ')
            print(gray_image_H_for_stat.shape)
        if count==1:
            all_gray_H_for_stat=gray_image_H_for_stat
            list_of_each_tile_H_mean=[np.mean(gray_image_H)]
            list_of_each_tile_H_std=[np.std(gray_image_H)]
        else:
            all_gray_H_for_stat=np.append(all_gray_H_for_stat, gray_image_H_for_stat, axis=0)
            list_of_each_tile_H_mean=np.append(list_of_each_tile_H_mean,[np.mean(gray_image_H)], axis=0)
            list_of_each_tile_H_std=np.append(list_of_each_tile_H_std,[np.std(gray_image_H)], axis=0)
        
        gray_image_E=np.dot(RGB_absorption,RGB_absorption_to_E)
        bool_E=gray_image_E>=E_PORTION_LOW_THRESH
        gray_image_E_for_stat=gray_image_E[bool_E]

        if count==1:
            all_gray_E_for_stat=gray_image_E_for_stat
            list_of_each_tile_E_mean=[np.mean(gray_image_E)]
            list_of_each_tile_E_std=[np.std(gray_image_E)]
        else:
            all_gray_E_for_stat=np.append(all_gray_E_for_stat, gray_image_E_for_stat, axis=0)
            list_of_each_tile_E_mean=np.append(list_of_each_tile_E_mean,[np.mean(gray_image_E)], axis=0)
            list_of_each_tile_E_std=np.append(list_of_each_tile_E_std,[np.std(gray_image_E)], axis=0)
        one_image_relative_path=f_input_image_list_txt.readline()
        one_image_relative_path=one_image_relative_path.replace('\n','')
        
    print('start to sort')
    sorted_gray_H=np.sort(all_gray_H_for_stat, axis=None)#flatten(force to 1D) and sort, ascending order
    total_qualified_H_num=sorted_gray_H.size#number of elements
    maxH=sorted_gray_H[int(total_qualified_H_num*(100.0-GET_TOP_PERCENT)/100.0)]

    sorted_gray_E=np.sort(all_gray_E_for_stat, axis=None)#flatten(force to 1D) and sort, ascending order
    total_qualified_E_num=sorted_gray_E.size#number of elements
    maxE=sorted_gray_E[int(total_qualified_E_num*(100.0-GET_TOP_PERCENT)/100.0)]

    mean_H_WSI=np.mean(list_of_each_tile_H_mean)/(maxH*ADDITIONAL_NORM_FACTOR)
    mean_E_WSI=np.mean(list_of_each_tile_E_mean)/(maxE*ADDITIONAL_NORM_FACTOR)
    std_H_WSI=np.mean(list_of_each_tile_H_std)/(maxH*ADDITIONAL_NORM_FACTOR)
    std_E_WSI=np.mean(list_of_each_tile_E_std)/(maxE*ADDITIONAL_NORM_FACTOR)

    print('maxH = {}, maxE = {}, mean_H_WSI_normalized_by_max= {}, mean_E_WSI_normalized_by_max= {}, std_H_WSI_normalized_by_max= {}, std_E_WSI_normalized_by_max= {}'.format(maxH,maxE,mean_H_WSI,mean_E_WSI,std_H_WSI,std_E_WSI))
    print('real max H = {}, max E ={}, real min H ={} min E = {}'.format(sorted_gray_H[-1],sorted_gray_E[-1],sorted_gray_H[0],sorted_gray_E[0]))
    f_track.write('maxH = {}, maxE = {}, mean_H_WSI_normalized_by_max= {}, mean_E_WSI_normalized_by_max= {}, std_H_WSI_normalized_by_max= {}, std_E_WSI_normalized_by_max= {}\n'.format(maxH,maxE,mean_H_WSI,mean_E_WSI,std_H_WSI,std_E_WSI))
    f_track.write('real max H = {}, max E ={}, real min H ={} min E = {}\n'.format(sorted_gray_H[-1],sorted_gray_E[-1],sorted_gray_H[0],sorted_gray_E[0]))

    return maxH,maxE,mean_H_WSI,mean_E_WSI,std_H_WSI,std_E_WSI


if __name__ == '__main__':
    print('Start to get normalization factors for each WSI based on maxH, maxE...')
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(f'{OUTPUT_FILE_NAME}', 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
            'absorption_R(7);absorption_G(8);absorption_B(9);'+\
            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30);H_E_angle_thresh(31);'+\
            'H_exclude_portion(32);E_exclude_portion(33);H_angle_degree(34);E_angle_degree(35);'+\
            'Matrix_RGBabsorption_to_HERes_r0Hc0R(36);r0Hc1G(37);r0Hc2B(38);r1Ec0R(39);r1Ec1G(40);r1Ec2B(41);r2RESc0R(42);r2RESc1G(43);r2RESc2B(44);'+\
            'H_max(45);E_max(46);H_mean_normalized_by_max(47);E_mean_normalized_by_max(48);H_std_normalized_by_max(49);E_std_normalized_by_max(50)\n')
        for row in range(0,TOTAL_NUM_WSIS):
            f.write('\n')
        f.close()
    with open(INPUT_MATRIX_INFO_TXT, 'r', encoding='utf-8') as input_file:
        txt_data = input_file.readlines()

    output_file=open(f'{OUTPUT_FILE_NAME}', 'r', encoding='utf-8')
    txt_output_data = output_file.readlines()
    output_file.close()

    f_track=open(OUTPUT_TRACK_PROCESS,'a+')
    f_track.write('Start to get eigen vectors/values of covariance matrix for each WSI...\n')

    for row in range(process_image_start,process_image_end+1):
        print('\nProcessing image in row '+str(row))
        f_track.write('\nProcessing image in row {}\n'.format(row))
        txt_input_line=txt_data[row]
        txt_input_line_split=txt_input_line.split(';')
        txt_input_line_split[-1]=txt_input_line_split[-1].replace('\n','')
        input_image_list_txt=INPUT_IMAGE_LIST_PATH+txt_input_line_split[1]+'_'+txt_input_line_split[3]+\
            '/20230207_'+txt_input_line_split[1]+'_'+txt_input_line_split[3]+\
            '_All_foreground_tiles_saved_tiles_list.txt'
        background_R=float(txt_input_line_split[4])
        background_G=float(txt_input_line_split[5])
        background_B=float(txt_input_line_split[6])
        background_RGB_matrix=[background_R,background_G,background_B]
        matrix_RGB2HE=[[float(txt_input_line_split[36]),float(txt_input_line_split[37]),float(txt_input_line_split[38])],\
                        [float(txt_input_line_split[39]),float(txt_input_line_split[40]),float(txt_input_line_split[41])]]
        maxH,maxE,mean_H_WSI,mean_E_WSI,std_H_WSI,std_E_WSI=get_normalization_factor_maxH_maxE_from_tiles_in_one_WSI(INPUT_IMAGE_DIR, input_image_list_txt, background_RGB_matrix, matrix_RGB2HE, f_track)
        if row == process_image_start:
            dataset_mean_H_list=[mean_H_WSI]
            dataset_mean_E_list=[mean_E_WSI]
            dataset_std_H_list=[std_H_WSI]
            dataset_std_E_list=[std_E_WSI]
        else:
            dataset_mean_H_list=np.append(dataset_mean_H_list, [mean_H_WSI], axis=0)
            dataset_mean_E_list=np.append(dataset_mean_E_list,[mean_E_WSI], axis=0)
            dataset_std_H_list=np.append(dataset_std_H_list, [std_H_WSI], axis=0)
            dataset_std_E_list=np.append(dataset_std_E_list, [std_E_WSI], axis=0)
        txt_output_line=txt_input_line.replace('\n',';'+str(maxH)+';'+str(maxE)+';'+str(mean_H_WSI)+';'+str(mean_E_WSI)+';'+str(std_H_WSI)+';'+str(std_E_WSI)+'\n')
        txt_output_data[row]=txt_output_line
        
        with open(f'{OUTPUT_FILE_NAME}', 'w', encoding='utf-8') as output_file:
            output_file.writelines(txt_output_data)
    dataset_mean_H=np.mean(dataset_mean_H_list)
    dataset_mean_E=np.mean(dataset_mean_E_list)
    dataset_std_H=np.mean(dataset_std_H_list)
    dataset_std_E=np.mean(dataset_std_E_list)
    print('Dataset level (after normalized by maxH, maxE): mean_H={}, mean_E={}, std_H={}, std_E={}'.format(dataset_mean_H,dataset_mean_E,dataset_std_H,dataset_std_E))
    input_file.close()
    f_track.write('Dataset level (after normalized by maxH, maxE): mean_H={}, mean_E={}, std_H={}, std_E={} \n'.format(dataset_mean_H,dataset_mean_E,dataset_std_H,dataset_std_E))
    f_track.write('Program end...\n\n\n\n\n\n\n')
    f_track.close()



        


