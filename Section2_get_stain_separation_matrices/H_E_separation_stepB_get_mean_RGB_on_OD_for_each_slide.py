# This step B code reads step A's output as input to get list of WSIs,
# and also reads all the foreground tiles in each WSI to get mean of RGB on OD space for each WSI. (absorption of RGB channel)
# You need to set set process_image_start=1, process_image_end=420, EXTRACT_TILES_OPTION=2, 
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

#speed: around 1 WSI/minute, it is faster on CHPC server than SCI server. 
import os
from PIL import Image
import numpy as np
TOTAL_NUM_WSIS=420#420 for TCGA
process_image_start=1
process_image_end=TOTAL_NUM_WSIS#included, for TCGA
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with X+1 lines. 
DOWNSAMPLE_RATE=10#downsample each tile
max_image_num_thresh=-1#-1(all), 5
GREEN_color_FIXED_thresh_max=255
RED_color_FIXED_thresh_min=0
OUTPUT_PATH=''#current folder
OUTPUT_FILE_NAME='StepB_output_mean_of_RGB_on_OD.txt'
INPUT_BACKGROUND_COLOR_TXT_PATH_NAME='StepA_output_background_color.txt'
INPUT_IMAGE_DIR='/your_own_path/tiles_20230301_1_20X/'
INPUT_IMAGE_LIST_PATH='/your_own_path/tiles_20230301_1_20X/All_foreground_tiles/'

def get_mean_RGB_on_OD_of_each_image_listed_in_txt(input_img_dir, input_image_list_txt, background_R, background_G, background_B):
    
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

        image_ones=np.ones(image_selected.shape,dtype=np.float32)
        image_selected = np.maximum(image_selected,image_ones)#change all pixel intensity with 0 value to 1
        
        RGB_OD_values_in_one_tile=np.log10((image_ones/image_selected)*np.array([background_R, background_G, background_B]))

        if count==1:
            mean_RGB_OD_values_in_WSI_tiles=RGB_OD_values_in_one_tile
        else:
            mean_RGB_OD_values_in_WSI_tiles=np.append(mean_RGB_OD_values_in_WSI_tiles, RGB_OD_values_in_one_tile, axis=0)

        if count>=max_image_num_thresh and max_image_num_thresh>0:
            break
        one_image_relative_path=f_input_image_list_txt.readline()
        one_image_relative_path=one_image_relative_path.replace('\n','')#necessary
    
    mean_RGB_OD_value_in_WSI=np.mean(mean_RGB_OD_values_in_WSI_tiles,axis=0)
    print('Mean OD_value in WSI:')
    print(mean_RGB_OD_value_in_WSI)
    f_input_image_list_txt.close()
    return mean_RGB_OD_value_in_WSI
    

if __name__ == '__main__':
    print('Start to get mean of R, G, B OD_value for each WSI...')
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(f'{OUTPUT_PATH}{OUTPUT_FILE_NAME}', 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6);'+\
                'OD_value_R(7);OD_value_G(8);OD_value_B(9)\n')
        for row in range(0,TOTAL_NUM_WSIS):
            f.write('\n')
        f.close()
    with open(INPUT_BACKGROUND_COLOR_TXT_PATH_NAME, 'r', encoding='utf-8') as input_file:
        txt_data = input_file.readlines()
    output_file=open(f'{OUTPUT_PATH}{OUTPUT_FILE_NAME}', 'r+', encoding='utf-8')
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
        
        mean_RGB_OD_value=get_mean_RGB_on_OD_of_each_image_listed_in_txt(INPUT_IMAGE_DIR, input_image_list_txt, background_R, background_G, background_B)
        txt_output_line=txt_input_line.replace('\n',';'+str(mean_RGB_OD_value[0])+';'+str(mean_RGB_OD_value[1])+';'+str(mean_RGB_OD_value[2])+'\n')
        txt_output_data[row]=txt_output_line
        
        with open(f'{OUTPUT_PATH}{OUTPUT_FILE_NAME}', 'w', encoding='utf-8') as output_file:
            output_file.writelines(txt_output_data)
    input_file.close()
    
    
