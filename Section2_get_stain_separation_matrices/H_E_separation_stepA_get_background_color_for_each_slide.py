# The step A code reads Whole Slide Images (WSIs) and outputs background color in R, G, B channels for each WSI. 
# Input:
# ORIGINAL_IMAGES_PATH is the path to the folder that saves all WSIs.
# excel_loc is the path to the Excel file that saves list of 420 TCGA WSIs.

# Output:
# OUTPUT_TXT_FILE_NAME saves txt file that contains background color in R, G, B channels. 
# The columns of output txt file are: 'index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6)'
# Each row saves each WSI's results.

# In this code, we set DIRECTLY_SET_255_AS_BACKGROUND_COLOR=True



#New: no need to calculate background color, directly use 255!
#1. Many pixel intensity larger than calculated background color
#To directly set 255 as background, enable DIRECTLY_SET_255_AS_BACKGROUND_COLOR=True

#speed: 10 WSIs/minutes
#stepA: get background color for each WSI. We divide WSI into 4 parts, each part, we sample many patches to get brightest patch. We get avg 
# among 2 median patches in 4 parts. 

import xlrd#excel
import slideio
import numpy as np

DIRECTLY_SET_255_AS_BACKGROUND_COLOR=True
FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING=True#if first time, we generate output txt file first with 441 lines. 
OUTPUT_TXT_FILE_NAME='StepA_output_background_color.txt'

TOTAL_NUM_WSIS_IN_EXCEL=420
SAMPLE_ROWS_COLS_IN_EACH_QUARTER=30#100
SAMPLE_PATCH_SIZE=21#
SAMPLE_ORIGINAL_PATCH_SIZE=21#If it is smaller than 21, it is possible that read_block returns out of boundary error

process_image_start=1
process_image_end=420#included
ORIGINAL_IMAGES_PATH='/your_own_path/Section1_get_tiles_from_WSIs/TCGA_ccRCC_WSIs/'# path to all WSIs
excel_loc = ('StepA_input_WSIs_list.xlsx')
OUTPUT_PATH=''#current folder

def get_RGB_background_color_from_WSI(WSI, height, width, sample_rows_cols):
    if DIRECTLY_SET_255_AS_BACKGROUND_COLOR:
        return 255.0, 255.0, 255.0
    else:
        if int(width/2/sample_rows_cols)<SAMPLE_ORIGINAL_PATCH_SIZE or \
            int(height/2/sample_rows_cols)<SAMPLE_ORIGINAL_PATCH_SIZE:
            sample_rows_cols=min(int(width/2/SAMPLE_ORIGINAL_PATCH_SIZE), int(height/2/SAMPLE_ORIGINAL_PATCH_SIZE))-1
            print('sample_rows_cols changed to {sample_rows_cols}')
        
        brightest_int_R=[0.0,0.0,0.0,0.0]
        brightest_int_G=[0.0,0.0,0.0,0.0]
        brightest_int_B=[0.0,0.0,0.0,0.0]
        for quarter_index in range(0,4):
            if quarter_index==0:
                region_start_x=0
                region_end_x=int(width/2)
                region_start_y=0
                region_end_y=int(height/2)
            elif quarter_index==1:
                region_start_x=int(width/2)
                region_end_x=width
                region_start_y=0
                region_end_y=int(height/2)
            elif quarter_index==2:
                region_start_x=0
                region_end_x=int(width/2)
                region_start_y=int(height/2)
                region_end_y=height
            else:
                region_start_x=int(width/2)
                region_end_x=width
                region_start_y=int(height/2)
                region_end_y=height
            brightest_intensity=0.0
            for sample_row in range(0,sample_rows_cols):
                for sample_col in range(0, sample_rows_cols):
                    patch_start_x=region_start_x+int(sample_col*width/2/sample_rows_cols)
                    patch_start_y=region_start_y+int(sample_row*height/2/sample_rows_cols)
                    patch_np=WSI.read_block((patch_start_x,patch_start_y,SAMPLE_ORIGINAL_PATCH_SIZE,SAMPLE_ORIGINAL_PATCH_SIZE),size=(SAMPLE_PATCH_SIZE,SAMPLE_PATCH_SIZE))
                    patch_np=patch_np.astype(float)
                    patch_color=np.mean(patch_np, axis=(0,1))
                    if patch_color[0]+patch_color[1]+patch_color[2]>brightest_intensity:
                        brightest_intensity=patch_color[0]+patch_color[1]+patch_color[2]
                        brightest_int_R[quarter_index]=patch_color[0].tolist()#convert numpy float to float
                        brightest_int_G[quarter_index]=patch_color[1].tolist()
                        brightest_int_B[quarter_index]=patch_color[2].tolist()
            print(f'RGB in quarter {quarter_index}:')
            print(brightest_int_R[quarter_index], brightest_int_G[quarter_index], brightest_int_B[quarter_index])
        R=max(brightest_int_R)
        G=max(brightest_int_G)
        B=max(brightest_int_B)
        return R, G, B


if __name__ == '__main__':
    #excel first row: index_row;index_name_WSI_relative_path;WSI_name
    #excel second row: 1;IMG0001;TCGA-CJ-4891-01Z-00-DX1.5E9F7E8F-110A-4A9F-BCD0-9CFBA14E1419.svs;TCGA-CJ-4891-01Z-00-DX1.5E9F7E8F-110A-4A9F-BCD0-9CFBA14E1419
    if FIRST_TIME_TO_RUN_OR_RERUN_EVERYTHING:
        f = open(OUTPUT_PATH+OUTPUT_TXT_FILE_NAME, 'w')
        f.write('index_row(0);index_name(1);WSI_relative_path(2);WSI_name(3);background_R(4);background_G(5);background_B(6)\n')
        for row in range(0,TOTAL_NUM_WSIS_IN_EXCEL):
            f.write('\n')
        f.close()
    excel_wb = xlrd.open_workbook(excel_loc)
    excel_sheet = excel_wb.sheet_by_index(0)
    print(excel_sheet.cell_value(1, 3))
    for row in range(process_image_start,process_image_end+1):
        image_relative_path_name_ext=excel_sheet.cell_value(row, 2)
        slide = slideio.open_slide(ORIGINAL_IMAGES_PATH+image_relative_path_name_ext,'SVS')
        num_scenes = slide.num_scenes
        scene = slide.get_scene(0)
        print('\nProcessing image in row '+str(row)+' in Excel...')
        print('num_scenes: {}'.format(num_scenes))
        print(scene.name, scene.rect, scene.num_channels)#1 Image (0, 0, 100720width, 84412height) 3
        scene_width=scene.rect[2]
        scene_height=scene.rect[3]
        background_R, background_G, background_B=get_RGB_background_color_from_WSI(scene, scene_height, scene_width, SAMPLE_ROWS_COLS_IN_EACH_QUARTER)
        print(f'Background color: {background_R}, {background_G}, {background_B}')
        with open(OUTPUT_PATH+OUTPUT_TXT_FILE_NAME, 'r', encoding='utf-8') as file:
            txt_data = file.readlines()
        
        txt_data[row] = str(excel_sheet.cell_value(row, 0))+';'+\
                        excel_sheet.cell_value(row, 1)+';'+\
                        excel_sheet.cell_value(row, 2)+';'+\
                        excel_sheet.cell_value(row, 3)+';'+\
                        str(background_R)+';'+\
                        str(background_G)+';'+\
                        str(background_B)+'\n'
                        
        
        with open(OUTPUT_PATH+OUTPUT_TXT_FILE_NAME, 'w', encoding='utf-8') as file:
            file.writelines(txt_data)





