# In step C code, we read tiles to get covariance matrix of RGB on OD space.

# Input:
# INPUT_MEAN_OF_RGB_OD_VALUE_TXT_PATH_NAME: output of step B
# INPUT_IMAGE_DIR: root directory that saves the tiles
# INPUT_IMAGE_LIST_PATH: directory of txt files that saves the list of tiles. Each txt saves one WSI's list of all foreground tiles.

# Output:
# OUTPUT_FILE_NAME: output file that saves the output of step B + covariance matrix info
# Columns in output file are index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);
#                OD_value_R(7);OD_value_G(8);OD_value_B(9);cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18)


#around 6 WSI/minute
import os
from PIL import Image
import numpy as np
TOTAL_NUM_WSIS=420
process_image_start=1
process_image_end=TOTAL_NUM_WSIS#included, for TCGA
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with X+1 lines. 
DOWNSAMPLE_RATE=10#downsample each tile
max_image_num_thresh=-1#-1(all), 5
GREEN_color_FIXED_thresh_max=255
RED_color_FIXED_thresh_min=0
OUTPUT_FILE_NAME='stepC_output_RGB_OD_value_cov_mat.txt'
INPUT_MEAN_OF_RGB_OD_VALUE_TXT_PATH_NAME='StepB_output_mean_of_RGB_on_OD.txt'
INPUT_IMAGE_DIR='/your_own_path/tiles_20230301_1_20X/'
INPUT_IMAGE_LIST_PATH='/your_own_path/tiles_20230301_1_20X/All_foreground_tiles/'

def get_RGB_OD_value_covariance_matrix_of_one_WSI_based_on_tiles(input_img_dir, input_image_list_txt, background_RGB,mean_OD_value_RGB):
    
    f_input_image_list_txt = open(input_image_list_txt, 'r')
    one_image_relative_path = f_input_image_list_txt.readline()
    one_image_relative_path=one_image_relative_path.replace('\n','')#necessary
    count=0

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

        image_ones=np.ones(image_selected.shape,dtype=np.float32)
        image_selected = np.maximum(image_selected,image_ones)#change all pixel intensity =0 to =1
        
        RGB_OD_values_in_one_tile=np.log10((image_ones/image_selected)*np.array(background_RGB))

        if count==1:
            RGB_OD_values_in_WSI_tiles=RGB_OD_values_in_one_tile
        else:
            RGB_OD_values_in_WSI_tiles=np.append(RGB_OD_values_in_WSI_tiles, RGB_OD_values_in_one_tile, axis=0)

        if count>=max_image_num_thresh and max_image_num_thresh>0:
            break
        one_image_relative_path=f_input_image_list_txt.readline()
        one_image_relative_path=one_image_relative_path.replace('\n','')#necessary

    print('Check RGB_OD_values_in_WSI_tiles shape:')
    print(RGB_OD_values_in_WSI_tiles.shape)
    num_pixels=RGB_OD_values_in_WSI_tiles.shape[0]
    mean_RGB_OD_value=np.asarray(mean_OD_value_RGB)
    RGB_OD_value_minus_mean=RGB_OD_values_in_WSI_tiles-mean_RGB_OD_value[None,:]
    cov_mat_one_WSI=np.matmul(RGB_OD_value_minus_mean.transpose(),RGB_OD_value_minus_mean)
    div_mat = num_pixels * np.ones([3,3])
    cov_mat_one_WSI = np.divide(cov_mat_one_WSI, div_mat)

    print('covariance matrix in WSI:')
    print(cov_mat_one_WSI)
    f_input_image_list_txt.close()
    return cov_mat_one_WSI
    

if __name__ == '__main__':
    print('Start to get R, G, B covariance matrix on OD space for each WSI...')
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(f'{OUTPUT_FILE_NAME}', 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
                'OD_value_R(7);OD_value_G(8);OD_value_B(9);cov_mat_r0c0(10);r0c1(11);r0c2(12);r1c0(13);r1c1(14);r1c2(15);r2c0(16);r2c1(17);r2c2(18)\n')
                #rxcy means the element at row x, col y
        for row in range(0,TOTAL_NUM_WSIS):
            f.write('\n')
        f.close()
    with open(INPUT_MEAN_OF_RGB_OD_VALUE_TXT_PATH_NAME, 'r', encoding='utf-8') as input_file:
        txt_data = input_file.readlines()
    output_file=open(f'{OUTPUT_FILE_NAME}', 'r', encoding='utf-8')
    txt_output_data = output_file.readlines()
    output_file.close()
    for row in range(process_image_start,process_image_end+1):
        print('Processing image in row '+str(row))
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
        
        cov_mat=get_RGB_OD_value_covariance_matrix_of_one_WSI_based_on_tiles(INPUT_IMAGE_DIR, input_image_list_txt, background_RGB,mean_OD_value_RGB)
        txt_output_line=txt_input_line.replace('\n',';'+str(cov_mat[0][0])+';'+str(cov_mat[0][1])+';'+str(cov_mat[0][2])+\
                                                    ';'+str(cov_mat[1][0])+';'+str(cov_mat[1][1])+';'+str(cov_mat[1][2])+\
                                                    ';'+str(cov_mat[2][0])+';'+str(cov_mat[2][1])+';'+str(cov_mat[2][2])+'\n')
        txt_output_data[row]=txt_output_line
        
        with open(f'{OUTPUT_FILE_NAME}', 'w', encoding='utf-8') as output_file:
            output_file.writelines(txt_output_data)
    input_file.close()
    
    