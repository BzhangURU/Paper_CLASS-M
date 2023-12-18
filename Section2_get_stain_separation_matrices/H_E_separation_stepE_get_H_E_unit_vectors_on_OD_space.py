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

import os
from PIL import Image
from numpy import linalg as LA
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
PI=3.14159265

H_exclude_portion=0.01
E_exclude_portion=0.01
total_epoch=1000
TOTAL_NUM_WSIS=420#420
process_image_start=1#min 1
process_image_end=TOTAL_NUM_WSIS#max TOTAL_NUM_WSIS, included, for TCGA
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with 441 lines. 
PREPROCESS_DATASET=True
if PREPROCESS_DATASET:
    preprocess_str='_dataset_preprocessed'
    MIN_DISTANCE_SQUARE_THRESH=0.0225#0.15*0.15 based on Macenko paper, delete pixels too close to background color
else:
    preprocess_str=''
str_show_settings='H_E_exclude_portion_'+'{:.2f}'.format(H_exclude_portion)+'_'+'{:.2f}'.format(E_exclude_portion)

OUTPUT_SUBFOLDER='StepE_output_training_record_'+str_show_settings+preprocess_str+'/'
OUTPUT_TRAIN_RECORD_PREFIX=OUTPUT_SUBFOLDER+'StepE_train_record_'
OUTPUT_TXT_NAME='StepE_output_H_E_angles_'+str_show_settings+preprocess_str+'.txt'
OUTPUT_TRACK_PROCESS='StepE_output_track_running_process_'+str_show_settings+preprocess_str+'.txt'

INPUT_EIGEN_VECTORS_VALUES_TXT_NAME='StepD_output_eigen_values_vectors_of_cov_mat.txt'
INPUT_ROOT_PATH='StepD_output_pixel_coordinates_on_OD/'
INPUT_PT_PREFIX=INPUT_ROOT_PATH+'StepD_output_coordinates_on_OD_'

def numbers_have_no_change(a, b):
    if a*b>=0 and 1.00000001*np.absolute(a)>np.absolute(b) and 1.00000001*np.absolute(b)>np.absolute(a):
        return True
    else:
        return False

def train_to_get_H_E_angle_on_one_WSI(row, input_dataset_path, output_train_record_txt):
    
    coordinates_dataset_original = torch.load(input_dataset_path)#currently loaded size is (2,num of pixels)
    if PREPROCESS_DATASET:
        #verified in just_practice.py and succeeded.
        coordinates_temp1=coordinates_dataset_original[0]
        coordinates_temp2=coordinates_dataset_original[1]
        coordinates_temp1=torch.square(coordinates_temp1)#squared
        coordinates_temp2=torch.square(coordinates_temp2)#squared
        coordinates_temp1=torch.add(coordinates_temp1,coordinates_temp2)
        bool_qualified=coordinates_temp1>=MIN_DISTANCE_SQUARE_THRESH
        coordinates_dataset=coordinates_dataset_original[:,bool_qualified]
    else:
        coordinates_dataset=coordinates_dataset_original
    num_total_samples=float(coordinates_dataset.size(dim=1))
    f_track_train_record_txt=open(output_train_record_txt, 'w')
    H_angle_wide_bound=torch.tensor(-90.0*PI/180, dtype=torch.float64)#ground truth H angle should be negative angle
    H_angle_narrow_bound=torch.tensor(90.0*PI/180, dtype=torch.float64)
    E_angle_wide_bound=torch.tensor(90.0*PI/180, dtype=torch.float64)
    E_angle_narrow_bound=torch.tensor(-90.0*PI/180, dtype=torch.float64)

    f_track_train_record_txt.write(f"Initial angle: H angle(degree) = {H_angle_wide_bound.numpy()*180.0/PI}  --  {H_angle_narrow_bound.numpy()*180.0/PI}, "+\
                                   f"E angle(degree) = {E_angle_narrow_bound.numpy()*180.0/PI}  --  {E_angle_wide_bound.numpy()*180.0/PI}\n")
    
    x_coordinates=coordinates_dataset[0]
    y_coordinates=coordinates_dataset[1]
    for epoch in range(total_epoch):
        H_angle_new=(H_angle_wide_bound+H_angle_narrow_bound)/2.0
        E_angle_new=(E_angle_wide_bound+E_angle_narrow_bound)/2.0

        bool_outside_H_new=y_coordinates<x_coordinates*torch.tan(H_angle_new)
        count_outside_H_new=bool_outside_H_new.sum().item()#sentence verified
        if count_outside_H_new>num_total_samples*H_exclude_portion:
            H_angle_narrow_bound=H_angle_new
        else:
            H_angle_wide_bound=H_angle_new

        bool_outside_E_new=y_coordinates>x_coordinates*torch.tan(E_angle_new)
        count_outside_E_new=bool_outside_E_new.sum().item()#sentence verified
        if count_outside_E_new>num_total_samples*E_exclude_portion:
            E_angle_narrow_bound=E_angle_new
        else:
            E_angle_wide_bound=E_angle_new

        print(f"image index(row)={row}, epoch={epoch}, \
H_angle_wide_bound(degree)={H_angle_wide_bound.numpy()*180/PI}, H_angle_narrow_bound(degree)={H_angle_narrow_bound.numpy()*180/PI}, \
E_angle_narrow_bound(degree)={E_angle_narrow_bound.numpy()*180/PI}, E_angle_wide_bound(degree)={E_angle_wide_bound.numpy()*180/PI}")

        f_track_train_record_txt.write(f"image index(row)={row}, epoch={epoch}, \
H_angle_wide_bound(degree)={H_angle_wide_bound.numpy()*180/PI}, H_angle_narrow_bound(degree)={H_angle_narrow_bound.numpy()*180/PI}, \
E_angle_narrow_bound(degree)={E_angle_narrow_bound.numpy()*180/PI}, E_angle_wide_bound(degree)={E_angle_wide_bound.numpy()*180/PI}\n")
        
        if numbers_have_no_change(H_angle_wide_bound.numpy(), H_angle_narrow_bound.numpy()) and \
            numbers_have_no_change(E_angle_narrow_bound.numpy(), E_angle_wide_bound.numpy()):
            f_track_train_record_txt.write(f"H_angle_new(degree)={H_angle_new.numpy()*180/PI},\
                                           E_angle_new(degree)={E_angle_new.numpy()*180/PI}\n")
            f_track_train_record_txt.write('Parameters no longer updated. Training Ended...\n\n\n\n\n\n')
            f_track_train_record_txt.close()
            print('Parameters no longer updated. Training Ended...')
            return H_angle_new.numpy(), E_angle_new.numpy()
        
    f_track_train_record_txt.write('Reached end of epochs. Program Ends...\n\n\n\n\n\n')
    f_track_train_record_txt.close()
    return H_angle_new.numpy(), E_angle_new.numpy()

if __name__ == '__main__':
    print('Start to get eigen vectors/values of covariance matrix for each WSI...')
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(f'{OUTPUT_TXT_NAME}', 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
            'OD_value_R(7);OD_value_G(8);OD_value_B(9);'+\
            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30);H_E_angle_thresh(31);'+\
            'H_exclude_portion(32);E_exclude_portion(33);H_angle_degree(34);E_angle_degree(35);'+\
            'Matrix_RGBOD_value_to_HERes_r0Hc0R(36);r0Hc1G(37);r0Hc2B(38);r1Ec0R(39);r1Ec1G(40);r1Ec2B(41);r2RESc0R(42);r2RESc1G(43);r2RESc2B(44)\n')
        for row in range(0,TOTAL_NUM_WSIS):
            f.write('\n')
        f.close()
    with open(INPUT_EIGEN_VECTORS_VALUES_TXT_NAME, 'r', encoding='utf-8') as input_file:
        txt_data = input_file.readlines()

    output_file=open(f'{OUTPUT_TXT_NAME}', 'r', encoding='utf-8')
    txt_output_data = output_file.readlines()
    output_file.close()

    f_track=open(OUTPUT_TRACK_PROCESS,'a+')
    f_track.write('Start training for each WSI...\n')

    isExist = os.path.exists(OUTPUT_SUBFOLDER)
    if not isExist:
        os.makedirs(OUTPUT_SUBFOLDER)
        print('New directory '+OUTPUT_SUBFOLDER+' is created!')
        f_track.write('New directory {} is created\n'.format(OUTPUT_SUBFOLDER))
    print(torch.__version__)
    for row in range(process_image_start,process_image_end+1):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('\n\nProcessing image in row '+str(row))
        f_track.write('\nProcessing image in row {}, current time={}\n'.format(row, current_time))
        txt_input_line=txt_data[row]
        txt_input_line_split=txt_input_line.split(';')
        txt_input_line_split[-1]=txt_input_line_split[-1].replace('\n','')
        input_dataset_path=INPUT_PT_PREFIX+txt_input_line_split[1]+'_'+txt_input_line_split[3]+'.pt'
        output_train_record_txt=OUTPUT_TRAIN_RECORD_PREFIX+txt_input_line_split[1]+'_'+txt_input_line_split[3]+'.txt'

        H_angle_radius_detach_np, E_angle_radius_detach_np=train_to_get_H_E_angle_on_one_WSI(row, input_dataset_path, output_train_record_txt)
        print('H E angle(degree):')
        print(H_angle_radius_detach_np*180/PI)
        print(E_angle_radius_detach_np*180/PI)
        f_track.write('H E angle(degree):')
        f_track.write(str(H_angle_radius_detach_np*180/PI))
        f_track.write('   ')
        f_track.write(str(E_angle_radius_detach_np*180/PI))

        v1_dominant=np.asarray([float(txt_input_line_split[19]), float(txt_input_line_split[20]), float(txt_input_line_split[21])])
        v2_subdominant=np.asarray([float(txt_input_line_split[22]), float(txt_input_line_split[23]), float(txt_input_line_split[24])])
        v3_residual=np.asarray([float(txt_input_line_split[25]), float(txt_input_line_split[26]), float(txt_input_line_split[27])])

        vH=np.cos(H_angle_radius_detach_np)*v1_dominant+np.sin(H_angle_radius_detach_np)*v2_subdominant
        vE=np.cos(E_angle_radius_detach_np)*v1_dominant+np.sin(E_angle_radius_detach_np)*v2_subdominant
        vH_2dim = np.expand_dims(vH, axis=1)
        vE_2dim = np.expand_dims(vE, axis=1)
        v3_2dim = np.expand_dims(v3_residual, axis=1)

        RM_HERes_to_RGBOD_value=np.concatenate((vH_2dim, vE_2dim, v3_2dim), axis=1)

        M_RGBOD_value_to_HERes=np.linalg.inv(RM_HERes_to_RGBOD_value)
        print('M_RGBOD_value_to_HERes')
        print(M_RGBOD_value_to_HERes)

        txt_output_line=txt_input_line.replace('\n',';'+'H_E_angle_thresh'+';'+str(H_exclude_portion)+';'+str(E_exclude_portion)+\
                                                    ';'+str(H_angle_radius_detach_np*180.0/PI)+';'+str(E_angle_radius_detach_np*180.0/PI)+\
                                                    ';'+str(M_RGBOD_value_to_HERes[0][0])+';'+str(M_RGBOD_value_to_HERes[0][1])+';'+str(M_RGBOD_value_to_HERes[0][2])+\
                                                    ';'+str(M_RGBOD_value_to_HERes[1][0])+';'+str(M_RGBOD_value_to_HERes[1][1])+';'+str(M_RGBOD_value_to_HERes[1][2])+\
                                                    ';'+str(M_RGBOD_value_to_HERes[2][0])+';'+str(M_RGBOD_value_to_HERes[2][1])+';'+str(M_RGBOD_value_to_HERes[2][2])+'\n')
        txt_output_data[row]=txt_output_line
        
        with open(f'{OUTPUT_TXT_NAME}', 'w', encoding='utf-8') as output_file:
            output_file.writelines(txt_output_data)
    input_file.close()
    f_track.write('Program end...\n\n\n\n\n\n\n')
    f_track.close()