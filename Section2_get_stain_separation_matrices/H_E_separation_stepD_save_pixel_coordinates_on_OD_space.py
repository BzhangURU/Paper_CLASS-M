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
# OUTPUT_TRACK_PROCESS: file that records running progress. 
# OUTPUT_FILE_NAME: output file that saves the output of step C + eigen vectors and eigen values of covariance matrices.
# Columns in output file are 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
#            'OD_value_R(7);OD_value_G(8);OD_value_B(9);'+\
#            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
#            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
#            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
#            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
#            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30)'

#with downsample_rate=10, 6 WSIs/minute
import os
from PIL import Image
from numpy import linalg as LA
import numpy as np
import torch

TOTAL_NUM_WSIS=420#420
process_image_start=1#min 1
process_image_end=420#max TOTAL_NUM_WSIS, included, for TCGA
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with all lines. 
DOWNSAMPLE_RATE=10#downsample each tile
max_image_num_thresh=-1#-1(all), 5
GREEN_color_FIXED_thresh_max=255
RED_color_FIXED_thresh_min=0

OUTPUT_SUBFOLDER='StepD_output_pixel_coordinates_on_OD/'
OUTPUT_PT_PREFIX=OUTPUT_SUBFOLDER+'StepD_output_coordinates_on_OD_'
OUTPUT_FILE_NAME='StepD_output_eigen_values_vectors_of_cov_mat.txt'
OUTPUT_TRACK_PROCESS='StepD_output_track_running_process.txt'

INPUT_RGB_ABSORPTION_COV_MAT_TXT_NAME='stepC_output_RGB_OD_value_cov_mat.txt'
INPUT_IMAGE_DIR='/your_own_path/tiles_20230301_1_20X/'
INPUT_IMAGE_LIST_PATH='/your_own_path/tiles_20230301_1_20X/All_foreground_tiles/'

def save_pixel_distribution_on_eigen_vectors_for_one_WSI(input_img_dir, input_image_list_txt, output_each_pixel_dist_on_plane_dataset_path, v1_x, v2_y, background_RGB, f_track):
    f_input_image_list_txt = open(input_image_list_txt, 'r')
    one_image_relative_path = f_input_image_list_txt.readline()
    one_image_relative_path=one_image_relative_path.replace('\n','')#necessary
    count=0
    v1_v2_mat=np.append(np.expand_dims(v1_x, axis=0),np.expand_dims(v2_y, axis=0),axis=0)

    while len(one_image_relative_path)>=3:
        count+=1
        cur_img_dir=input_img_dir+one_image_relative_path
        image_original = np.asarray(Image.open(os.path.join(cur_img_dir)))#(height, width, channel)
        image_original = image_original.astype('float32')
        image_downsampled=image_original[::DOWNSAMPLE_RATE,::DOWNSAMPLE_RATE,:]
        image_reshaped=np.reshape(image_downsampled,(-1,3))
        image_Red=image_reshaped[:,0]#Image.open returns RGB, openCV returns BGR
        image_Green=image_reshaped[:,1]
        bool_Red=image_Red>=RED_color_FIXED_thresh_min
        bool_Green=image_Green<=GREEN_color_FIXED_thresh_max
        bool_select=bool_Red & bool_Green
        image_selected=image_reshaped[bool_select]
        if count==1:
            print('Check first tile shape: ')
            print(image_selected.shape)
            f_track.write('Check first tile shape: \n')
            f_track.write(str(image_selected.shape[0]))
            f_track.write('\t')
            f_track.write(str(image_selected.shape[1]))
            f_track.write('\n')

        image_ones=np.ones(image_selected.shape,dtype=np.float32)
        image_selected = np.maximum(image_selected,image_ones)#change all pixel intensity =0 to =1
        
        RGB_OD_values_in_one_tile=np.log10((image_ones/image_selected)*np.array(background_RGB))#shape (n,3)
        RGB_OD_values_in_one_tile_transpose=np.transpose(RGB_OD_values_in_one_tile)#shape (3,n)

        if count==1:
            RGB_OD_values_in_one_WSI_transpose=RGB_OD_values_in_one_tile_transpose
        else:#shape (3,N)
            RGB_OD_values_in_one_WSI_transpose=np.append(RGB_OD_values_in_one_WSI_transpose, RGB_OD_values_in_one_tile_transpose, axis=1)#axis=1!!!

        if count>=max_image_num_thresh and max_image_num_thresh>0:
            break
        one_image_relative_path=f_input_image_list_txt.readline()
        one_image_relative_path=one_image_relative_path.replace('\n','')#necessary

    WSI_v1_v2_coordinates=np.matmul(v1_v2_mat,RGB_OD_values_in_one_WSI_transpose)
    WSI_v1_v2_coordinates_tensor=torch.from_numpy(WSI_v1_v2_coordinates)#(2,3)x(3,num_pixels)=(2,num_pixels)

    print('Check WSI_v1_v2_coordinates shape:')
    print(WSI_v1_v2_coordinates.shape)
    print('Check WSI_v1_v2_coordinates first element:')
    print(WSI_v1_v2_coordinates[:,0])
    f_track.write('Check WSI_v1_v2_coordinates shape:\n')
    f_track.write(str(WSI_v1_v2_coordinates.shape[0]))
    f_track.write('\t')
    f_track.write(str(WSI_v1_v2_coordinates.shape[1]))
    f_track.write('\n')
    f_track.write('Check WSI_v1_v2_coordinates first element:\n')
    f_track.write("\t".join(str(item) for item in WSI_v1_v2_coordinates[:,0]))
    f_track.write('\n')

    f_input_image_list_txt.close()
    torch.save(WSI_v1_v2_coordinates_tensor, output_each_pixel_dist_on_plane_dataset_path)

if __name__ == '__main__':
    print('Start to get eigen vectors/values of covariance matrix for each WSI...')
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(f'{OUTPUT_FILE_NAME}', 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
            'OD_value_R(7);OD_value_G(8);OD_value_B(9);'+\
            'cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18);'+\
            'eigen_vector_dominant_element0(19);eigen_vector_dominant_element1(20);eigen_vector_dominant_element2(21);'+\
            'eigen_vector_sub_dominant_element0(22);eigen_vector_sub_dominant_element1(23);eigen_vector_sub_dominant_element2(24);'+\
            'eigen_vector_residual_element0(25);eigen_vector_residual_element1(26);eigen_vector_residual_element2(27);'+\
            'eigen_value_dominant(28);eigen_value_sub_dominant(29);eigen_value_residual(30)\n')
        for row in range(0,TOTAL_NUM_WSIS):
            f.write('\n')
        f.close()
    with open(INPUT_RGB_ABSORPTION_COV_MAT_TXT_NAME, 'r', encoding='utf-8') as input_file:
        txt_data = input_file.readlines()

    output_file=open(f'{OUTPUT_FILE_NAME}', 'r', encoding='utf-8')
    txt_output_data = output_file.readlines()
    output_file.close()

    f_track=open(OUTPUT_TRACK_PROCESS,'a+')
    f_track.write('Start to get eigen vectors/values of covariance matrix for each WSI...\n')

    isExist = os.path.exists(OUTPUT_SUBFOLDER)
    if not isExist:
        os.makedirs(OUTPUT_SUBFOLDER)
        print('New directory '+OUTPUT_SUBFOLDER+' is created!')
        f_track.write('New directory {} is created\n'.format(OUTPUT_SUBFOLDER))

    for row in range(process_image_start,process_image_end+1):
        print('Processing image in row '+str(row))
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
        background_RGB=[background_R,background_G,background_B]
        mean_OD_value_R=float(txt_input_line_split[7])
        mean_OD_value_G=float(txt_input_line_split[8])
        mean_OD_value_B=float(txt_input_line_split[9])
        mean_OD_value_RGB=[mean_OD_value_R,mean_OD_value_G,mean_OD_value_B]
        mean_OD_value_RGB_np=np.asarray(mean_OD_value_RGB)

        cov_mat=np.asarray([[float(txt_input_line_split[10]), float(txt_input_line_split[11]), float(txt_input_line_split[12])],
                    [float(txt_input_line_split[13]), float(txt_input_line_split[14]), float(txt_input_line_split[15])],
                    [float(txt_input_line_split[16]), float(txt_input_line_split[17]), float(txt_input_line_split[18])]])
        eigen_values_ascending_order, eigen_vectors_check_each_column = LA.eigh(cov_mat)

        print('eigen vectors, eigen values')
        print(eigen_vectors_check_each_column)
        print(eigen_values_ascending_order)
        f_track.write('eigen vectors, eigen values\n')
        #f_track.write({eigen_vectors_check_each_column})
        for i in eigen_vectors_check_each_column:
            f_track.write('\t'.join(str(j) for j in i) + '\n' )
        f_track.write('\n')
        f_track.write("\t".join(str(item) for item in eigen_values_ascending_order))
        f_track.write('\n')
        #if not, we should switch the two vectors to make sure H-E vectors is around dominant axis. 
        #Based on test, this would not happen. 
        assert np.absolute(np.dot(mean_OD_value_RGB_np,eigen_vectors_check_each_column[:,2]))>\
            np.absolute(np.dot(mean_OD_value_RGB_np,eigen_vectors_check_each_column[:,1]))

        #the dominant vector elements should be all positive, otherwise all negative
        if eigen_vectors_check_each_column[0][2]<0.0:
            eigen_vectors_check_each_column[0][2]=-eigen_vectors_check_each_column[0][2]
            eigen_vectors_check_each_column[1][2]=-eigen_vectors_check_each_column[1][2]
            eigen_vectors_check_each_column[2][2]=-eigen_vectors_check_each_column[2][2]

        assert eigen_vectors_check_each_column[0][2]>0.0 and eigen_vectors_check_each_column[1][2]>0.0 and \
            eigen_vectors_check_each_column[1][2]>0.0

        #The red channel on sub-dominant vector should be negative, then H-angle would be in quadrant IV, E-angle in quadrant I.
        #Because compare to H, red has more relative portion in Eosin. 
        if eigen_vectors_check_each_column[0][1]>0.0:
            eigen_vectors_check_each_column[0][1]=-eigen_vectors_check_each_column[0][1]
            eigen_vectors_check_each_column[1][1]=-eigen_vectors_check_each_column[1][1]
            eigen_vectors_check_each_column[2][1]=-eigen_vectors_check_each_column[2][1]

        output_each_pixel_dist_on_plane_dataset_path=OUTPUT_PT_PREFIX+txt_input_line_split[1]+'_'+txt_input_line_split[3]+'.pt'
        v1_x=np.asarray(eigen_vectors_check_each_column[:,2])
        v2_y=np.asarray(eigen_vectors_check_each_column[:,1])

        #make sure cross product of v1_x and v2_y(same direction as v_H x v_E because we made sure H-angle is negative) has same direction as v_residual!
        cross_product_v1_x_v2_y=np.cross(v1_x,v2_y)
        if np.dot(cross_product_v1_x_v2_y,np.asarray(eigen_vectors_check_each_column[:,0]))<0:
            eigen_vectors_check_each_column[0,0]=-eigen_vectors_check_each_column[0,0]
            eigen_vectors_check_each_column[1,0]=-eigen_vectors_check_each_column[1,0]
            eigen_vectors_check_each_column[2,0]=-eigen_vectors_check_each_column[2,0]
        v3_residual=np.asarray(eigen_vectors_check_each_column[:,0])

        #assert last column is all positive(or all negative->all negative), assert center column red is negative
        save_pixel_distribution_on_eigen_vectors_for_one_WSI(INPUT_IMAGE_DIR, input_image_list_txt, output_each_pixel_dist_on_plane_dataset_path, v1_x, v2_y, background_RGB, f_track)
    
        txt_output_line=txt_input_line.replace('\n',';'+str(v1_x[0])+';'+str(v1_x[1])+';'+str(v1_x[2])+\
                                                    ';'+str(v2_y[0])+';'+str(v2_y[1])+';'+str(v2_y[2])+\
                                                    ';'+str(v3_residual[0])+';'+str(v3_residual[1])+';'+str(v3_residual[2])+\
            ';'+str(eigen_values_ascending_order[2])+';'+str(eigen_values_ascending_order[1])+';'+str(eigen_values_ascending_order[0])+'\n')
        txt_output_data[row]=txt_output_line
        
        with open(f'{OUTPUT_FILE_NAME}', 'w', encoding='utf-8') as output_file:
            output_file.writelines(txt_output_data)
    input_file.close()
    f_track.close()