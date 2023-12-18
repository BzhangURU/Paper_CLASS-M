# This is part of our work for CLASS-M model. 
# This code is an example of our code for cropping tiles based on polygon annotations for TCGA WSIs. 
# Related work is introduced in https://arxiv.org/abs/2312.06978
# Please cite our work if you want to use the code to generate tiles from WSIs. 

#This code use whole slide images and annotated polygons as input, and will 
#generate tiles inside polygons and other visualization images. 
#to run the code, search the date "2023xxxx", then change the date to prevent overwritting

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,37).__str__()
import numpy as np # Do not forget to import numpy
import determine_tile_inside_polygon
import matplotlib.pyplot as plt
import cv2
import json
import xlrd#read excel. The latest version of xlrd(2.01) only supports .xls files. Install the older version 1.2.0 to open .xlsx files.   
#import xlwt
from datetime import datetime
import slideio
#to get tiles in polygons, set tile_save_limit_in_each_polygon, process_image_start=271, EXTRACT_TILES_OPTION=1, SAVE_PRESENTATION_PATH, TILES_OUTPUT_ROOT_PATH
#to get all tiles, set tile_save_limit_in_each_polygon, process_image_start=1, EXTRACT_TILES_OPTION=2, SAVE_PRESENTATION_PATH, TILES_OUTPUT

#frequent changes
num_images=420
tile_save_limit_in_each_polygon=2000000   #3, 2000000  max saved number of tiles in a polygon in step3
#1-270 unlabeled training set, 271-300 labeled training set, 301-360 validation set, 361-420 test set
process_image_start=271        #set index of image you start cropping from, 1 is smallest, num_images is largest
process_image_end=300        #set index of image you end up cropping from, 1 is smallest, num_images is largest, the end is contained to be processed
ENABLE_STEP1=True
ENABLE_STEP2=False#Keep it False!!!
ENABLE_STEP3=True
EXTRACT_TILES_OPTION=2 #1: extract foreground tiles inside polygons,  2: extract all foreground tiles everywhere in WSIs,   
#3: extract all foreground tiles outside of all polygons. So 1 U 3 = 2
#if EXTRACT_TILES_OPTION==1, OUTSIDE_POLYGON_TYPE doesn't matter
#OUTSIDE_POLYGON_TYPE='Outside_all_polygons'#for EXTRACT_TILES_OPTION=3: extract all foreground tiles outside of all polygons.
OUTSIDE_POLYGON_TYPE='All_foreground_tiles'#for EXTRACT_TILES_OPTION=2: extract all foreground tiles everywhere in WSIs,  
#output, folders will be auto-created
SAVE_PRESENTATION_PATH='presentation_images_20230301_1_polygons_20X/'#all_tiles, polygons
TILES_OUTPUT_ROOT_PATH='tiles_20230301_1_20X/'

#less frequent changes
GREEN_color_FIXED_thresh_max=225
GREEN_color_FIXED_thresh_max_for_fat=240
RED_color_FIXED_thresh_min=100#70
CHANGE_SPACE_TO_UNDERLINE=True#Example: change "High NC" to "High_NC"
INITIALIZE_LIST_WITH_SORTED_GP=True#use step_0_get_list_of_GP_names.py to get initialize_GP_list()
Polygon_Too_Large_Thresh0_1=1.0#1.0, 0.25
SAVING_TILE_SIZE=400
SAVING_TILE_JUMP_STEP=400#in code, tiny polygons would automatically have smaller steps
TILE_RESO_MUL=2#tile resolution, original is 40X(set to 1), if desire tile with 20X, set to 2, if 10X, set to 4
USE_OPENCV_TO_READ_IMAGE=False
image_name_start_index_in_path=len('/data/images/tcga/kirc/')#len('/data/images/tcga/kirc/')
count_image_limit=10000000

#input
#Section1_get_tiles_from_WSIs
ORIGINAL_IMAGES_PATH='TCGA_ccRCC_WSIs/'
ANNOTATION_FILE_PATH='polygon_annotations_of_WSIs/polygon_annotation_details/'
excel_loc = ("polygon_annotations_of_WSIs/summary_of_polygon_annotations.xlsx")
#
# ORIGINAL_IMAGES_PATH='/home/sci/bodong.zhang/projects/2021_kidney_cancer/Deepika_original_images/'
# ANNOTATION_FILE_PATH='/home/sci/bodong.zhang/projects/2021_kidney_cancer/Deepika_annotations/overall/'
# excel_loc = ("/home/sci/bodong.zhang/projects/2021_kidney_cancer/Deepika_annotations/overall/manifest.xlsx")

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Tile:
    def __init__(self, x, y, dx, dy):
        self.x = x#start point
        self.y = y#start point
        self.dx = dx#side length
        self.dy = dy#side length

class Polygon:
    def __init__(self,GPname,normalized_points_list_of_list):
        self.GPname=GPname#growth pattern name
        self.nor_points_lofl=normalized_points_list_of_list

#GP: growth pattern
def process_this_GP(GPname):
    return True

#We can get this from step_0_get_list_of_GP_names.py
def initialize_GP_list():
    if EXTRACT_TILES_OPTION==1:
        return [['Acinar', 0], ['Alveolar', 0], ['EN', 0], ['Fat', 0], ['FN', 0], ['Gleason-3+3', 0], ['Gleason-4+4', 0], \
        ['Gleason-benign', 0], ['Hemangiomatous', 0], ['High_NC', 0], ['LGS', 0], ['Lymph-Positive', 0], ['Lymphocytes', 0], \
            ['Necrosis', 0], ['Normal', 0], ['Oncocytic_', 0], ['Paneth_cell', 0], ['Papillary', 0], ['Pleomorphic_giant_cells', 0], \
                ['Prostate-Gleason3', 0], ['Prostate-Gleason4', 0], ['Prostate-Gleason5', 0], ['Regressive', 0], ['Rhabdoid', 0], \
                    ['Sarc', 0], ['SN', 0], ['SSP', 0], ['Tubular', 0], ['Tubulopapillary', 0]]#TCGA
    else:
        return [[OUTSIDE_POLYGON_TYPE,0]]

def this_GP_should_have_subfolder(GPname):
    return True

def save_this_tile(GPname, index):
    return process_this_GP(GPname)

def get_random_color_based_on_GP(name):
    if len(name)<2:
        return (0,0,0)
    elif len(name)==2:
        return ((ord(name[0])*17)%125, (ord(name[1])*17)%125, (ord(name[0])*11)%125)#BGR
    else:
        return ((ord(name[0])*17)%125, (ord(name[1])*17)%125, (ord(name[2])*17)%125)#BGR

#detect dark background, shallow black ink, green ink and get rid of them
def find_gray_green_color(avgIntR, avgIntG, avgIntB):
    #to disable this function, directly return False
    if avgIntR>=RED_color_FIXED_thresh_min and avgIntG<GREEN_color_FIXED_thresh_max:#This condition is also checked outside of function.
        if avgIntR<130 and avgIntG<130 and avgIntB<130:
            if avgIntG*100>80*avgIntR and avgIntG*100>80*avgIntB:
                return True
        elif avgIntR<150 and avgIntG<150 and avgIntB<150:
            if avgIntG*100>88*avgIntR and avgIntG*100>88*avgIntB:
                return True
        elif avgIntR<200 and avgIntG<200 and avgIntB<200:
            if avgIntG*100>90*avgIntR and avgIntG*100>90*avgIntB:
                return True
        elif avgIntG*100>96*avgIntR and avgIntG*100>96*avgIntB:
            return True
    return False


def get_positive_region(image_name):
    #Some WSIs have ink areas, vague areas or copied regions, so we get rid of those areas.
    nor_positive_region=Tile(0.0, 0.0, 1.0, 1.0)#start_x(width), start_y, dx, dy, original point is topleft
    if image_name.find('TCGA-CZ-5459-01Z-00-DX1.300c3ae5-9e88-4336-9807-a61cd71d4f1b')>=0:#'key_words'
        nor_positive_region=Tile(0.0, 0.4, 1.0, 0.291)#positive region for all TILES_OPTION==1,2,3!!!!!!!!
    if EXTRACT_TILES_OPTION==2 or EXTRACT_TILES_OPTION==3:
        if image_name.find('TCGA-CJ-4891-01Z-00-DX1.5E9F7E8F-110A-4A9F-BCD0-9CFBA14E1419')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.58, 1.0)
        if image_name.find('TCGA-AS-3778-01Z-00-DX1.E44DEEAC-DDAC-476A-A8EB-A7E3689B4121')>=0:#'key_words'
            nor_positive_region=Tile(0.33, 0.0, 0.33, 1.0)
        if image_name.find('TCGA-B0-4700-01Z-00-DX1.1a56e8e5-e1c7-48c7-b17a-5adc9d780613')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.48, 1.0)
        if image_name.find('TCGA-DV-5567-01Z-00-DX1.646edc51-54e3-4999-8554-07f724b906a6')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-5551-01Z-00-DX1.eafbba65-d1ae-49e2-9253-5de17ebcb713')>=0:#'key_words'
            nor_positive_region=Tile(0.6, 0.0, 0.4, 1.0)
        if image_name.find('TCGA-B2-5633-01Z-00-DX1.c55a29ca-468e-46de-9ff5-a51f46fe09fd')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.6, 1.0, 0.4)
        if image_name.find('TCGA-BP-5169-01Z-00-DX1.b9739d1e-6d86-44ab-a064-9fe50d393613')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-BP-4763-01Z-00-DX1.6f3d1ec1-6404-42a9-9668-2cf0ac931a46')>=0:#'key_words'
            nor_positive_region=Tile(0.22, 0.19, 0.27, 0.38)
        if image_name.find('TCGA-CJ-4869-01Z-00-DX1.0963305A-0BA8-4266-A4D7-B37A98FA342F')>=0:#'key_words'
            nor_positive_region=Tile(0.38, 0.29, 0.32, 0.39)
        if image_name.find('TCGA-BP-5170-01Z-00-DX1.ae43bef7-3d81-4f69-be37-b4958bf79939')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 1.0, 0.85)
        if image_name.find('TCGA-CZ-5466-01Z-00-DX1.f0fd74a8-2430-45dd-9c76-8794619ccf9d')>=0:#'key_words'
            nor_positive_region=Tile(0.027, 0.0, 0.48, 1.0)
        if image_name.find('TCGA-BP-4341-01Z-00-DX1.095921d7-ee43-4837-9dd4-263fe071ddb8')>=0:#'key_words'
            nor_positive_region=Tile(0.17, 0.0, 0.16, 1.0)
        if image_name.find('TCGA-B0-5102-01Z-00-DX1.0dc0a4bf-4836-4f9f-a31e-c03eb6855cf7')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.35, 1.0, 0.65 )
        if image_name.find('TCGA-B8-A54G-01Z-00-DX1.0EDB6818-907C-4C37-8645-628850202AD0')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-CJ-4889-01Z-00-DX1.A584930A-586C-4CD9-980E-FB0240AD5D27')>=0:#'key_words'
            nor_positive_region=Tile(0.524, 0.162, 0.175, 0.243)
        if image_name.find('TCGA-B0-4843-01Z-00-DX1.22597d5c-43e9-4ed1-a2c2-68d2f70fd5c4')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.189, 1.0, 0.811)
        if image_name.find('TCGA-B0-5698-01Z-00-DX1.e7739945-c890-46f9-ba2b-f2cd2ae39567')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-A54J-01Z-00-DX1.D9DA4491-BC23-4AA4-92CB-26956C78A034')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.4, 1.0)
        if image_name.find('TCGA-CJ-4899-01Z-00-DX1.D37DAF6F-9F3A-4A11-B4B6-37169AB592D9')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.35, 1.0)
        if image_name.find('TCGA-B8-5553-01Z-00-DX1.1fd7ba5f-3a88-49f7-b904-c17797092897')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.3, 1.0)
        if image_name.find('TCGA-CJ-6027-01Z-00-DX1.ADA07C08-C97A-4F41-9467-8F4F49F96EDA')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.35, 1.0)
        if image_name.find('TCGA-B8-A54I-01Z-00-DX1.76B5A794-746B-47DC-A771-3B56F24EF28B')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B2-5639-01Z-00-DX1.96ee3f26-65d8-4c4a-ad5b-eb396627f5bb')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.35, 1.0)
        if image_name.find('TCGA-B8-A54H-01Z-00-DX1.7DE86788-D09C-471E-B62D-2597EEAC3326')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B0-5096-01Z-00-DX1.badbc354-928e-4f7c-9b3f-341154bafbac')>=0:#'key_words'
            nor_positive_region=Tile(0.169, 0.333, 0.831, 0.667)
        if image_name.find('TCGA-B8-A54F-01Z-00-DX1.7C46D299-64F5-42C0-8226-A1AB4FF5C879')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-CJ-4874-01Z-00-DX1.614D8D7D-E9FA-464D-A90F-C5C789664287')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 1.0, 0.66)
        if image_name.find('TCGA-B2-5635-01Z-00-DX1.04222a5c-a2b2-4206-8a54-66c7474982d8')>=0:#'key_words'
            nor_positive_region=Tile(0.349, 0.0, 0.241, 1.0)
        if image_name.find('TCGA-BP-4973-01Z-00-DX1.2f753bb4-d7f4-43fc-994a-f076e6d10c50')>=0:#'key_words'
            nor_positive_region=Tile(0.223, 0.150, 0.663, 0.625)
        if image_name.find('TCGA-B8-5550-01Z-00-DX1.f9c84121-b555-486b-9049-9b87c7268e0a')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        # if image_name.find('TCGA-CZ-5455-01Z-00-DX1.f62e1d32-8a1e-4708-a0ba-93516a0b3dab')>=0:#'key_words'
        #     nor_positive_region=Tile(0.692, 0.773, 0.308, 0.227)
        # if image_name.find('TCGA-BP-5202-01Z-00-DX1.d7e64bbb-d747-4a01-a5d6-26ad5a1e8a13')>=0:#'key_words'
        #     nor_positive_region=Tile(0.0, 0.0, 0.134, 1.0)
        if image_name.find('TCGA-B8-A7U6-01Z-00-DX1.43BCA7B2-823E-4F38-BCF0-865E0FFF892A')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.45, 1.0)
        if image_name.find('TCGA-B8-5158-01Z-00-DX1.bf87760d-1ce2-4df9-a1a5-74f90d668ef2')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-DV-5565-01Z-00-DX1.41c51988-2bbf-4745-af8e-d45788995fca')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-4153-01Z-00-DX1.55bba0a1-79c2-475d-ae98-81a17d832ba6')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B2-4102-01Z-00-DX1.07c9ab3f-27ad-4e12-86e8-0eed36eb30c1')>=0:#'key_words'
            nor_positive_region=Tile(0.477, 0.0, 0.239, 1.0)
        if image_name.find('TCGA-B8-5552-01Z-00-DX1.391C74B0-7381-40F8-88AF-B1687CB39F11')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-BP-4762-01Z-00-DX1.d7e04665-32ee-4e53-a398-7ef13ab6630c')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-BP-5007-01Z-00-DX1.3d9c2588-3310-4fbc-8bc0-7185e9b0d83b')>=0:#'key_words'
            nor_positive_region=Tile(0.539, 0.0, 0.461, 1.0)
        if image_name.find('TCGA-A3-3316-01Z-00-DX1.d6248cf8-7b02-4dce-8c91-1b91751cf1ee')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 0.95)
        if image_name.find('TCGA-B8-4146-01Z-00-DX1.0F0EECE2-3ECC-4CEB-AAAC-DCE2AAF8E7CA')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-4148-01Z-00-DX1.ac30d91e-389c-4d6e-ae5a-048fd8a7a3bf')>=0:#'key_words'
            nor_positive_region=Tile(0.33, 0.0, 0.33, 1.0)
        if image_name.find('TCGA-BP-4798-01Z-00-DX1.5f6fa5de-bdbc-4048-9948-c89a25a92fd5')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.077, 0.92, 0.923)
        if image_name.find('TCGA-B8-5163-01Z-00-DX1.7e349a4e-7d00-4fc0-a754-52322827906c')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-A3-3358-01Z-00-DX1.1bd1c720-f6db-4837-8f83-e7476dd2b0a3')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B2-4099-01Z-00-DX1.c0aa462a-083d-4db8-af8f-37b45a544c8b')>=0:#'key_words'
            nor_positive_region=Tile(0.504, 0.0, 0.239, 1.0)
        if image_name.find('TCGA-CJ-4901-01Z-00-DX1.F720CF55-2204-4222-A99B-165FF257AF22')>=0:#'key_words'
            nor_positive_region=Tile(0.292, 0.0, 0.159, 1.0)
        if image_name.find('TCGA-B2-5641-01Z-00-DX1.f8058caa-63f7-421d-9953-c060913c3404')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 0.5)
        if image_name.find('TCGA-B2-A4SR-01Z-00-DX1.CC86884F-7784-495E-AFA0-FE57E4E6BADF')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-A8YJ-01Z-00-DX1.31A6E902-8704-4241-9BC0-EBD64309DEE7')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.45, 1.0)
        if image_name.find('TCGA-A3-3349-01Z-00-DX1.206c2817-1b93-4fdf-8128-3f6e3b4935b6')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-CJ-4907-01Z-00-DX1.12558019-745F-4EDF-A0A2-3228B475379F')>=0:#'key_words'
            nor_positive_region=Tile(0.091, 0.449, 0.705, 0.270)
        # if image_name.find('TCGA-B0-4816-01Z-00-DX1.6cc0b334-b885-4cdc-a2a8-cd3efaa2e241')>=0:#'key_words'
        #     nor_positive_region=Tile(0.0, 0.0, 0.453, 1.0)
        if image_name.find('TCGA-BP-4337-01Z-00-DX1.5c88395f-317b-4cc5-8945-16c7cdb0876e')>=0:#'key_words'
            nor_positive_region=Tile(0.05, 0.0, 0.95, 1.0)
        if image_name.find('TCGA-B0-4823-01Z-00-DX1.9a23a831-7fc7-404c-b195-b3b999b8dfe2')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.924, 0.925)
        if image_name.find('TCGA-B0-4842-01Z-00-DX1.d780158b-81c2-4ac9-b7a0-9c9386c6414c')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 1.0, 0.95)
        if image_name.find('TCGA-CJ-4920-01Z-00-DX1.103EF882-1E2D-443D-AEBC-EFC39CD5376F')>=0:#'key_words'
            nor_positive_region=Tile(0.421, 0.32, 0.164, 0.165)
        if image_name.find('TCGA-B0-4945-01Z-00-DX1.590b650c-c9cb-4601-886c-fde0ccd9b90d')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.189, 1.0, 0.811)
        if image_name.find('TCGA-B0-5113-01Z-00-DX1.db0ceba9-8e15-4308-bd78-dc587f5606a0')>=0:#'key_words'
            nor_positive_region=Tile(0.05, 0.0, 0.95, 1.0)
        if image_name.find('TCGA-BP-4330-01Z-00-DX1.9d3bec1f-552e-4515-b467-828b8f76503d')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.037, 1.0, 0.963)
        if image_name.find('TCGA-B2-3924-01Z-00-DX1.751b1107-fa18-4df5-89ae-42d06caf9b74')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-BP-4959-01Z-00-DX1.2f8b4ccb-c280-48e3-b901-50ef83e56669')>=0:#'key_words'
            nor_positive_region=Tile(0.142, 0.126, 0.639, 0.425)
        if image_name.find('TCGA-B8-5165-01Z-00-DX1.d0314896-5fb0-4c33-9d62-710199657d25')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-CJ-4886-01Z-00-DX1.14E6C96C-BD32-4EB4-8C9A-760A62D8DFDE')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 1.0, 0.224)
        if image_name.find('TCGA-CZ-5469-01Z-00-DX1.5c3baf64-ab33-4b32-b5e7-b7e4f42b6051')>=0:#'key_words'
            nor_positive_region=Tile(0.5, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-5549-01Z-00-DX1.96d1ead6-37ad-4e5d-bc95-2ef4fd1a940d')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-A54D-01Z-00-DX1.DCD17375-918C-4AEA-96F0-7FF8FCF4D00B')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-B8-A54D-01Z-00-DX1.DCD17375-918C-4AEA-96F0-7FF8FCF4D00B')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 1.0)
        if image_name.find('TCGA-BP-4354-01Z-00-DX1.9f53f9e2-ea72-4be4-9848-1a3f5634dac0')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.5, 1.0, 0.5)
        if image_name.find('TCGA-CZ-5462-01Z-00-DX1.270f84aa-9ae5-4de7-adb3-8b3ea53aeba1')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 0.95)
        if image_name.find('TCGA-B0-4693-01Z-00-DX1.1e9a856a-1172-4db8-887a-b65fff644798')>=0:#'key_words'
            nor_positive_region=Tile(0.32, 0.0, 0.68, 1.0)
        if image_name.find('TCGA-3Z-A93Z-01A-01-TSA.3B7A8AD8-E35C-420E-A819-097F34A52F54')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.4, 1.0)
        if image_name.find('TCGA-CJ-5686-01Z-00-DX1.9DC91DB0-272B-485C-9F61-7F7C00915746')>=0:#'key_words'
            nor_positive_region=Tile(0.353, 0.0, 0.371, 1.0)
        if image_name.find('TCGA-BP-4332-01Z-00-DX1.cce78224-1325-42ee-9211-b5142e0e3175')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.63, 1.0, 0.37)
        if image_name.find('TCGA-BP-5184-01Z-00-DX1.b682d92b-fc7b-4a8c-a0f3-d1e3a309f136')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.5, 1.0, 0.5)
        if image_name.find('TCGA-BP-4771-01Z-00-DX1.7b5ab331-81fe-4688-91ad-851a44191841')>=0:#'key_words'
            nor_positive_region=Tile(0.157, 0.0, 0.843, 1.0)
        if image_name.find('TCGA-B8-5162-01Z-00-DX1.3e999548-f3b9-46ca-b51a-d20735e1249b')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 0.95)
        if image_name.find('TCGA-B8-5159-01Z-00-DX1.12a9fb4e-2062-4e69-bd04-61eb2c1a7501')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 0.5, 0.95)
        if image_name.find('TCGA-CJ-4892-01Z-00-DX1.7732DE46-3F0B-4B38-A352-552F6F8A5CF3')>=0:#'key_words'
            nor_positive_region=Tile(0.3, 0.0, 0.7, 1.0)
        if image_name.find('TCGA-B0-4849-01Z-00-DX1.979ac1bc-d04f-470d-bed2-85d3bd3cd912')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.0, 1.0, 0.549)
        if image_name.find('TCGA-B0-5121-01Z-00-DX1.3e7c6f3c-998a-465f-a464-c301e71f2a89')>=0:#'key_words'
            nor_positive_region=Tile(0.0, 0.276, 0.651, 0.354)
        if image_name.find('TCGA-CJ-4893-01Z-00-DX1.056F80C3-B340-465A-B2A7-0523CEEA0219')>=0:#'key_words'
            nor_positive_region=Tile(0.625, 0.348, 0.219, 0.355)
    return nor_positive_region

#This is optional function, without it, we are still able to crop tiles in designated region.
#This function will rule out remaining regions early to increase speed.
def get_nor_polygon_from_positive_region(image_name):
    nor_positive_region=get_positive_region(image_name)
    positive_region_polygon=[   [nor_positive_region.x, nor_positive_region.y],\
                                [nor_positive_region.x, nor_positive_region.y+nor_positive_region.dy],\
                                [nor_positive_region.x+nor_positive_region.dx, nor_positive_region.y+nor_positive_region.dy],\
                                [nor_positive_region.x+nor_positive_region.dx, nor_positive_region.y]  ]
    return positive_region_polygon


def name_is_from_desired_set(full_name):
    return True

def tile_is_in_positive_region(tile, image_width_at_tile_resolution, image_height_at_tile_resolution, image_name):
    nor_positive_region=get_positive_region(image_name)
    tile_normalized_width_x=float(tile.x+tile.dx/2)/float(image_width_at_tile_resolution)
    tile_normalized_height_y=float(tile.y+tile.dy/2)/float(image_height_at_tile_resolution)
    if tile_normalized_width_x >= nor_positive_region.x and tile_normalized_width_x <= nor_positive_region.x+nor_positive_region.dx and \
        tile_normalized_height_y >= nor_positive_region.y and tile_normalized_height_y <= nor_positive_region.y+nor_positive_region.dy:
        return True
    else:
        return False

#Initialize count list!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#The above should be checked carefully and tuned in different tasks. 
##################################################################################################################
def create_subfolder_name(number, name):
    return 'IMG'+str(number).zfill(4)+'_'+name

def swap_columns(list_of_list, pos1, pos2):
    for one_list in list_of_list:
        one_list[pos1], one_list[pos2] = one_list[pos2], one_list[pos1]

def GP_index_in_saved_list(list_count_tiles, GP_name):
    #[['SN',5],['FN',],[...]]
    for index in range(len(list_count_tiles)):
        if GP_name==list_count_tiles[index][0]:
            return index
    return -1

#new algorithm to determine: sample 10x10 dots in each grid(each tile has multiple grids)
def tile_is_foreground_by_sampling_dots(imgXXX, tileXXX, GPname, image_name, force_to_work_on_openCV=False):
    if USE_OPENCV_TO_READ_IMAGE or force_to_work_on_openCV:
        if not tile_is_in_positive_region(tileXXX, imgXXX.shape[1], imgXXX.shape[0], image_name):
            return False
        #in here, general_img is whole picture
        general_img=imgXXX
        general_tile=tileXXX
    else:
        #img is slideio here
        #we only use this when we use slideio to read image and only when step 3
        if not tile_is_in_positive_region(tileXXX, imgXXX.rect[2]/TILE_RESO_MUL, imgXXX.rect[3]/TILE_RESO_MUL, image_name):
            return False
        tile_img_np=imgXXX.read_block((TILE_RESO_MUL*tileXXX.x,TILE_RESO_MUL*tileXXX.y,TILE_RESO_MUL*tileXXX.dx,TILE_RESO_MUL*tileXXX.dy),size=(tileXXX.dx,tileXXX.dy))#resize original WSI
        #in here(slideio), general_img is a tile. 
        general_img = cv2.cvtColor(np.array(tile_img_np), cv2.COLOR_RGB2BGR)#original tile_img
        general_tile=Tile(0,0,tileXXX.dx,tileXXX.dy)#original slideio_tile

#The following should not have imgXXX, tileXXX variables since we are using general general_img, general_tile for both openCV and slideio cases. 
    grid_dim=4#analyze/divide tile into grid_dim x grid_dim grids

    count_valid_grid=0
    count_valid_grid_thresh=13#13 out of 16 grids should be forground grid
    sampling_matrix_side_length=10#sample 10x10 dots in each grid
    G_color_thresh_max=GREEN_color_FIXED_thresh_max#225
    R_color_thresh_min=RED_color_FIXED_thresh_min#70
    if GPname=='Fat':
        G_color_thresh_max=GREEN_color_FIXED_thresh_max_for_fat#240
    grid_side_length_x=int(general_tile.dx/grid_dim)
    if grid_side_length_x==0:
        grid_side_length_x=1
    grid_side_length_y=int(general_tile.dy/grid_dim)
    if grid_side_length_y==0:
        grid_side_length_y=1
    sampling_step_x=int((grid_side_length_x)/(sampling_matrix_side_length))
    if sampling_step_x==0:
        sampling_step_x=1
    sampling_step_y=int((grid_side_length_y)/(sampling_matrix_side_length))
    if sampling_step_y==0:
        sampling_step_y=1
    for grid_index_x in range(grid_dim):
        for grid_index_y in range(grid_dim):
            count_sampling_dots=0
            avgIntG=0
            avgIntR=0
            avgIntB=0

            count_foreground_sampling_dots=0
            avgForegroundIntG=0
            avgForegroundIntR=0
            avgForegroundIntB=0
            for x in range(general_tile.x+grid_side_length_x*grid_index_x, general_tile.x+(grid_side_length_x*(grid_index_x+1)),sampling_step_x):
                for y in range(general_tile.y+grid_side_length_y*grid_index_y, general_tile.y+(grid_side_length_y*(grid_index_y+1)),sampling_step_y):
                    currentG=int(general_img[y][x][1])
                    currentR=int(general_img[y][x][2])
                    currentB=int(general_img[y][x][0])

                    avgIntG+=currentG
                    avgIntR+=currentR
                    avgIntB+=currentB
                    count_sampling_dots+=1

                    if currentG<=GREEN_color_FIXED_thresh_max and currentR>=R_color_thresh_min:
                        avgForegroundIntG+=currentG
                        avgForegroundIntR+=currentR
                        avgForegroundIntB+=currentB
                        count_foreground_sampling_dots+=1
            avgIntG=int(avgIntG/count_sampling_dots)   
            avgIntR=int(avgIntR/count_sampling_dots)
            avgIntB=int(avgIntB/count_sampling_dots)

            if count_foreground_sampling_dots>0:
                avgForegroundIntG=int(avgForegroundIntG/count_foreground_sampling_dots)
                avgForegroundIntR=int(avgForegroundIntR/count_foreground_sampling_dots)
                avgForegroundIntB=int(avgForegroundIntB/count_foreground_sampling_dots)
            #G_color_thresh_max could be GREEN_color_FIXED_thresh_max or GREEN_color_FIXED_thresh_max_for_fat
            if avgIntG<=G_color_thresh_max and avgIntR>=R_color_thresh_min:
                if not find_gray_green_color(avgForegroundIntR, avgForegroundIntG, avgForegroundIntB):
                    count_valid_grid+=1
            if avgIntR<R_color_thresh_min:
                return False
            if count_valid_grid>=count_valid_grid_thresh:
                break
        if count_valid_grid>=count_valid_grid_thresh:
            break
    if count_valid_grid>=count_valid_grid_thresh:
        return True
    else:
        return False

# points is list of Point with range (0,1)
def draw_a_polygon(img, normalized_points_list_of_list, name):
    color=get_random_color_based_on_GP(name)

    nor_x_min=1.0
    nor_x_max=0.0
    nor_y_min=1.0
    nor_y_max=0.0
    for i in range(len(normalized_points_list_of_list)):
        if nor_x_min>normalized_points_list_of_list[i][0]:
            nor_x_min=normalized_points_list_of_list[i][0]
        if nor_x_max<normalized_points_list_of_list[i][0]:
            nor_x_max=normalized_points_list_of_list[i][0]
        if nor_y_min>normalized_points_list_of_list[i][1]:
            nor_y_min=normalized_points_list_of_list[i][1]
        if nor_y_max<normalized_points_list_of_list[i][1]:
            nor_y_max=normalized_points_list_of_list[i][1]
    start_x=int(nor_x_min*img.shape[1])
    start_y=int(nor_y_min*img.shape[0])
    end_x=int(nor_x_max*img.shape[1])
    end_y=int(nor_y_max*img.shape[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (start_x, start_y)
    fontScale = 2
    txt_color = (0, 0, 0) #BGR
    thickness = 2

    line_thickness=4
    for i in range(len(normalized_points_list_of_list)):
        if i == len(normalized_points_list_of_list)-1:
            next_i=0
        else:
            next_i=i+1
        cv2.arrowedLine(img, (int(normalized_points_list_of_list[i][0]*img.shape[1]), int(normalized_points_list_of_list[i][1]*img.shape[0])), \
        (int(normalized_points_list_of_list[next_i][0]*img.shape[1]), int(normalized_points_list_of_list[next_i][1]*img.shape[0])), \
        color, thickness=line_thickness)
    return img

def tile_outside_annotation_polygons(data_json, tile, width, height):
    for item_json in data_json:
        geo_type=item_json['geometries']['features'][0]['geometry']['type']
        if geo_type!='Point':
            GPname=item_json['properties']['annotations']['name']
            #if GPname in GP_dict:
            normalized_points_list_of_list=item_json['geometries']['features'][0]['geometry']['coordinates'][0]
            nor_polygon=Polygon(GPname,normalized_points_list_of_list)
            polygon_Points=[]
            for i in range(len(nor_polygon.nor_points_lofl)):
                # polygon_Points.append(Point(img.shape[1]*nor_polygon.nor_points_lofl[i][0],img.shape[0]*nor_polygon.nor_points_lofl[i][1]))
                polygon_Points.append(Point(width*nor_polygon.nor_points_lofl[i][0],height*nor_polygon.nor_points_lofl[i][1]))
            if determine_tile_inside_polygon.tile_CENTER_is_inside_polygon(polygon_Points, tile):
                return False
    return True

#polygon.nor_points_lofl is normalized list of list
#This function is for resized image
def step1_1_draw_tiles_inside_a_polygon(resized_img_for_drawing, resized_img_for_scanning, data_json, nor_polygon, tile_size, tile_sample_step, polygon_index, GPname, image_index_and_name,\
    original_img, scale_percent, count_tiles_roughly, count_tiles_roughly_in_one_image, file_track_running_process):
    color=get_random_color_based_on_GP(GPname)
    line_thickness=2
    nor_x_min=1.0
    nor_x_max=0.0
    nor_y_min=1.0
    nor_y_max=0.0
    polygon_too_large=0
    for i in range(len(nor_polygon.nor_points_lofl)):
        if nor_x_min>nor_polygon.nor_points_lofl[i][0]:
            nor_x_min=nor_polygon.nor_points_lofl[i][0]
        if nor_x_max<nor_polygon.nor_points_lofl[i][0]:
            nor_x_max=nor_polygon.nor_points_lofl[i][0]
        if nor_y_min>nor_polygon.nor_points_lofl[i][1]:
            nor_y_min=nor_polygon.nor_points_lofl[i][1]
        if nor_y_max<nor_polygon.nor_points_lofl[i][1]:
            nor_y_max=nor_polygon.nor_points_lofl[i][1]
    if (nor_x_max-nor_x_min)*(nor_y_max-nor_y_min)>Polygon_Too_Large_Thresh0_1:#changable
        polygon_too_large=1
    start_x=int(nor_x_min*resized_img_for_scanning.shape[1])
    start_y=int(nor_y_min*resized_img_for_scanning.shape[0])
    end_x=int(nor_x_max*resized_img_for_scanning.shape[1])
    end_y=int(nor_y_max*resized_img_for_scanning.shape[0])

    polygon_Points=[]
    for i in range(len(nor_polygon.nor_points_lofl)):
        polygon_Points.append(Point(resized_img_for_scanning.shape[1]*nor_polygon.nor_points_lofl[i][0],resized_img_for_scanning.shape[0]*nor_polygon.nor_points_lofl[i][1]))

    if end_x-start_x<2*tile_size and end_y-start_y<2*tile_size:
        tile_sample_step=int(tile_size/4)
    elif (end_x-start_x<3*tile_size and end_y-start_y<4*tile_size) or (end_y-start_y<3*tile_size and end_x-start_x<4*tile_size):
        tile_sample_step=int(tile_size/2)

    if tile_sample_step==0:
            tile_sample_step=1
    tile_index=0
    if polygon_too_large==0:
        for x in range(start_x, end_x+1-tile_size, tile_sample_step):
            for y in range(start_y, end_y+1-tile_size, tile_sample_step):
                tile=Tile(x,y,tile_size,tile_size)
                tile_for_verifying_inside_polygon=Tile(x,y,tile_size,tile_size)
                if (end_x-start_x)*2<3*tile_size and (end_y-start_y)*2<3*tile_size:
                    tile_for_verifying_inside_polygon=Tile(x+int(tile_size/4),y+int(tile_size/4),int(tile_size/2),int(tile_size/2))
                elif (end_x-start_x<4*tile_size and end_y-start_y<6*tile_size) or (end_y-start_y<4*tile_size and end_x-start_x<6*tile_size):
                    tile_for_verifying_inside_polygon=Tile(x+int(tile_size/5),y+int(tile_size/5),int(3*tile_size/5),int(3*tile_size/5))
                if determine_tile_inside_polygon.tile_is_inside_polygon(polygon_Points, tile_for_verifying_inside_polygon):
                    if tile_is_foreground_by_sampling_dots(resized_img_for_scanning, tile, GPname, image_index_and_name, True):
                        if EXTRACT_TILES_OPTION!=3 or tile_outside_annotation_polygons(data_json,tile, resized_img_for_scanning.shape[1], resized_img_for_scanning.shape[0]):
                            cv2.rectangle(resized_img_for_drawing, (x,y), (x+tile_size,y+tile_size), color, line_thickness)
                            if (tile_index<1000 and (tile_index&255)==0) or (tile_index>1000 and (tile_index&1023)==0):
                                print('                Drawing tile '+str(tile_index)+'x, y: '+str(x)+', '+str(y)+'  s_x:'+str(start_x)+'  e_x:'+str(end_x)\
                                +'  s_y:'+str(start_y)+'  e_y:'+str(end_y)+'  tile_step:'+str(tile_sample_step))
                                file_track_running_process.write('                Drawing tile {}, x,y: {},{}, s_x:{}, e_x:{}, s_y:{}, e_y:{}\n'.format(tile_index,x,y,start_x, end_x, start_y, end_y))
                            tile_index+=1
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (start_x, start_y-5)
        if EXTRACT_TILES_OPTION>=2:
            if GPname==OUTSIDE_POLYGON_TYPE:
                pos = (20, 50)
        fontScale = 1
        font_color = get_random_color_based_on_GP(GPname) #BGR
        thickness = 2
        resized_img_for_drawing = cv2.putText(resized_img_for_drawing, 'No.'+str(polygon_index)+' '+GPname+' '+str(tile_index)+' tiles roughly, (size,step):'+str(tile_size)+' ,'+str(tile_sample_step), pos, font, fontScale, font_color, thickness, cv2.LINE_AA)
        GP_index=GP_index_in_saved_list(count_tiles_roughly, GPname)
        if GP_index<0:
            count_tiles_roughly_in_one_image.append([GPname,tile_index])
            count_tiles_roughly.append([GPname,tile_index])
        else:
            count_tiles_roughly_in_one_image[GP_index][1]+=tile_index
            count_tiles_roughly[GP_index][1]+=tile_index
    else:
        print('        Polygon too large, skip counting in resized image.')
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (start_x, start_y-5)
        fontScale = 1
        color = (0, 0, 0) #BGR
        thickness = 2
        resized_img_for_drawing = cv2.putText(resized_img_for_drawing, 'No.'+str(polygon_index)+' '+GPname+' (size,step):'+str(tile_size)+' ,'+str(tile_sample_step), pos, font, fontScale, color, thickness, cv2.LINE_AA)
    return resized_img_for_drawing

def step1_draw_tiles_polygons_inside_image(resized_img_for_drawing, resized_img_for_scanning, data_json, tile_size, tile_sample_step, image_index_and_name,image_relative_path, \
    img, scale_percent, count_tiles_roughly, count_tiles_roughly_in_one_image, file_track_running_process):
    # Iterating through the json
    polygon_index=0

    if EXTRACT_TILES_OPTION>=2:
        GPname=OUTSIDE_POLYGON_TYPE
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('\n        (In step 1)Now working on drawing tiles in WSIs: {}, current time={}\n'.format(OUTSIDE_POLYGON_TYPE, current_time))
        file_track_running_process.write('\n        (In step 1)Now working on drawing tiles in WSIs, current time={}\n'.format(polygon_index, GPname, current_time))
        #nor_polygon=Polygon(OUTSIDE_POLYGON_TYPE,[[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
        nor_polygon=Polygon(OUTSIDE_POLYGON_TYPE,get_nor_polygon_from_positive_region(image_index_and_name))
        resized_img_for_drawing=step1_1_draw_tiles_inside_a_polygon(resized_img_for_drawing, resized_img_for_scanning, data_json, nor_polygon, tile_size, tile_sample_step, polygon_index, \
            GPname, image_index_and_name, img, scale_percent, count_tiles_roughly, count_tiles_roughly_in_one_image, file_track_running_process)
    
    for item_json in data_json:
        geo_type=item_json['geometries']['features'][0]['geometry']['type']
        if geo_type!='Point':#Must be polygon, not point annotation to extract tiles. 
            GPname=item_json['properties']['annotations']['notes']
            if CHANGE_SPACE_TO_UNDERLINE:
                GPname=GPname.replace(" ", "_")
            #print(item)
            
            normalized_points_list_of_list=item_json['geometries']['features'][0]['geometry']['coordinates'][0]
            if len(normalized_points_list_of_list)>=3:#some annotations are single points, not polygons
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print('\n        (In step 1)Now processing polygon {}: {}, current time={}\n'.format(polygon_index, GPname, current_time))
                file_track_running_process.write('\n        (In step 1)Now processing polygon {}: {}, current time={}\n'.format(polygon_index, GPname, current_time))
                
                #Drawing polygons will affect resized presentation image's foreground decision if you draw polygon and then crop tiles. 
                #So we split into resized_img_for_drawing and resized_img_for_scanning.
                resized_img_for_drawing=draw_a_polygon(resized_img_for_drawing, normalized_points_list_of_list, GPname)
                nor_polygon=Polygon(GPname,normalized_points_list_of_list)
                if EXTRACT_TILES_OPTION==1:
                    resized_img_for_drawing=step1_1_draw_tiles_inside_a_polygon(resized_img_for_drawing, resized_img_for_scanning, data_json, nor_polygon, tile_size, tile_sample_step, polygon_index, \
                        GPname, image_index_and_name, img, scale_percent, count_tiles_roughly, count_tiles_roughly_in_one_image, file_track_running_process)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    nor_x_min=1.0
                    nor_y_min=1.0
                    for i in range(len(nor_polygon.nor_points_lofl)):
                        if nor_x_min>nor_polygon.nor_points_lofl[i][0]:
                            nor_x_min=nor_polygon.nor_points_lofl[i][0]
                        if nor_y_min>nor_polygon.nor_points_lofl[i][1]:
                            nor_y_min=nor_polygon.nor_points_lofl[i][1]
                    start_x=int(nor_x_min*resized_img_for_scanning.shape[1])
                    start_y=int(nor_y_min*resized_img_for_scanning.shape[0])
                    pos = (start_x, start_y-5)#x, y
                    fontScale = 1
                    font_color = get_random_color_based_on_GP(GPname) #BGR
                    thickness = 1
                    resized_img_for_drawing = cv2.putText(resized_img_for_drawing, 'No.'+str(polygon_index)+' '+GPname+'  (size,step):'+str(tile_size)+' ,'+str(tile_sample_step), pos, font, fontScale, font_color, thickness, cv2.LINE_AA)
                    
                polygon_index+=1

    cv2.imwrite(SAVE_PRESENTATION_PATH+'/20230207_'+image_index_and_name+'_resized_drawed.png', resized_img_for_drawing)

def step2_1_draw_a_polygon_boundary_image_with_tiles(img, normalized_points_list_of_list, GPname, image_index_and_name, polygon_index, tile_size, tile_sample_step,GP_count_tiles):
    nor_x_min=1.0
    nor_x_max=0.0
    nor_y_min=1.0
    nor_y_max=0.0
    for i in range(len(normalized_points_list_of_list)):
        if nor_x_min>normalized_points_list_of_list[i][0]:
            nor_x_min=normalized_points_list_of_list[i][0]
        if nor_x_max<normalized_points_list_of_list[i][0]:
            nor_x_max=normalized_points_list_of_list[i][0]
        if nor_y_min>normalized_points_list_of_list[i][1]:
            nor_y_min=normalized_points_list_of_list[i][1]
        if nor_y_max<normalized_points_list_of_list[i][1]:
            nor_y_max=normalized_points_list_of_list[i][1]
    start_x=int(nor_x_min*img.shape[1])
    start_y=int(nor_y_min*img.shape[0])
    end_x=int(nor_x_max*img.shape[1])
    end_y=int(nor_y_max*img.shape[0])

    polygon_boundary_img=img[start_y:end_y+1, start_x:end_x].copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    #pos = (120, 120)
    fontScale = 3
    txt_color=get_random_color_based_on_GP(GPname)#(0,0,0)
    color = get_random_color_based_on_GP(GPname) #(255, 0, 0) #BGR
    txt_thickness = 4
    #polygon_boundary_img = cv2.putText(polygon_boundary_img, GPname, pos, font, fontScale, txt_color, txt_thickness, cv2.LINE_AA)
    line_thickness=4
    for i in range(len(normalized_points_list_of_list)):
        if i == len(normalized_points_list_of_list)-1:
            next_i=0
        else:
            next_i=i+1
        cv2.line(polygon_boundary_img, (int(normalized_points_list_of_list[i][0]*img.shape[1]-start_x), \
        int(normalized_points_list_of_list[i][1]*img.shape[0]-start_y)), \
        (int(normalized_points_list_of_list[next_i][0]*img.shape[1]-start_x), \
        int(normalized_points_list_of_list[next_i][1]*img.shape[0]-start_y)), \
        color, thickness=line_thickness)

    polygon_Points=[]
    for i in range(len(normalized_points_list_of_list)):
        polygon_Points.append(Point(img.shape[1]*normalized_points_list_of_list[i][0]-start_x, \
        img.shape[0]*normalized_points_list_of_list[i][1]-start_y))

    tile_index=0
    #color=(255,255,0)
    pos=(120,120)
    
    line_thickness=3

    for x in range(0, end_x+1-start_x-tile_size, tile_sample_step):
        for y in range(0, end_y+1-start_y-tile_size, tile_sample_step):
            tile=Tile(x,y,tile_size,tile_size)
            if determine_tile_inside_polygon.tile_is_inside_polygon(polygon_Points, tile):
                if tile_is_foreground_by_sampling_dots(img, Tile(start_x+x,start_y+y,tile_size,tile_size), GPname, image_index_and_name):
                    cv2.rectangle(polygon_boundary_img, (x,y), (x+tile_size,y+tile_size), color, line_thickness)
                    if (tile_index<1000 and (tile_index&255)==0) or (tile_index>1000 and (tile_index&1023)==0):
                        print('                Drawing tile in a polygon boundary image '+str(tile_index)+'x, y: '+str(x)+', '+str(y)+\
                        '  e_x:'+str(end_x+1-start_x-tile_size)\
                        +'  e_y:'+str(end_y+1-start_y-tile_size)+'  tile_step:'+str(tile_sample_step))
                    tile_index+=1
    polygon_boundary_img = cv2.putText(polygon_boundary_img, 'No.'+str(polygon_index)+' '+GPname+' '+str(tile_index)+' tiles, (size,step):'+str(tile_size)+' ,'+str(tile_sample_step), pos, font, fontScale, txt_color, txt_thickness, cv2.LINE_AA)
    return polygon_boundary_img, tile_index

def step2_draw_polygon_boundary_images_and_their_tiles(img, data_json, tile_size, tile_sample_step, image_index_and_name, \
image_relative_path,GP_count_tiles):
    #The boundary of polygon would be boundary of image
    print('        Drawing polygon images...')

    file1 = open(SAVE_PRESENTATION_PATH+image_relative_path\
    +'/20230207_step2_'+image_index_and_name+'_count_tiles.txt','a+')
    
    polygon_index=0
    num_tiles=0
    for item_json in data_json:
        geo_type=item_json['geometries']['features'][0]['geometry']['type']
        if geo_type!='Point':
            GPname=item_json['properties']['annotations']['notes']
            if CHANGE_SPACE_TO_UNDERLINE:
                GPname=GPname.replace(" ", "_" )
            if process_this_GP(GPname):
                print('        Drawing polygon boundary image '+str(polygon_index)+' with GP: '+GPname)
                normalized_points_list_of_list=item_json['geometries']['features'][0]['geometry']['coordinates'][0]
                polygon_boundary_img, num_tiles=step2_1_draw_a_polygon_boundary_image_with_tiles(img, normalized_points_list_of_list, \
                GPname, image_index_and_name, polygon_index, tile_size, tile_sample_step,GP_count_tiles)
                cv2.imwrite(SAVE_PRESENTATION_PATH+image_relative_path\
                +'/20230207_'+image_index_and_name+'_polygon'+str(polygon_index)+'_'+GPname+'.png', polygon_boundary_img)
                if GPname in GP_count_tiles:
                    GP_count_tiles[GPname]+=num_tiles
                else:
                    GP_count_tiles[GPname]=num_tiles
                file1.write('        Polygon '+str(polygon_index)+' '+GPname+': '+str(num_tiles)+' tiles.\n')
                polygon_index+=1
    file1.close()

def step3_save_tiles_inside_a_polygon(img, data_json, nor_polygon, tile_size, tile_sample_step, image_index_and_name, GPname, polygon_index,\
    count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image, \
    count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image, file_track_running_process):
    nor_x_min=1.0
    nor_x_max=0.0
    nor_y_min=1.0
    nor_y_max=0.0
    for i in range(len(nor_polygon.nor_points_lofl)):
        if nor_x_min>nor_polygon.nor_points_lofl[i][0]:
            nor_x_min=nor_polygon.nor_points_lofl[i][0]
        if nor_x_max<nor_polygon.nor_points_lofl[i][0]:
            nor_x_max=nor_polygon.nor_points_lofl[i][0]
        if nor_y_min>nor_polygon.nor_points_lofl[i][1]:
            nor_y_min=nor_polygon.nor_points_lofl[i][1]
        if nor_y_max<nor_polygon.nor_points_lofl[i][1]:
            nor_y_max=nor_polygon.nor_points_lofl[i][1]
    
    if USE_OPENCV_TO_READ_IMAGE:
        start_x=int(nor_x_min*img.shape[1])
        start_y=int(nor_y_min*img.shape[0])
        end_x=int(nor_x_max*img.shape[1])
        end_y=int(nor_y_max*img.shape[0])
        desired_reso_img_width=img.shape[1]
        desired_reso_img_height=img.shape[0]

        polygon_Points=[]
        for i in range(len(nor_polygon.nor_points_lofl)):
            polygon_Points.append(Point(img.shape[1]*nor_polygon.nor_points_lofl[i][0],img.shape[0]*nor_polygon.nor_points_lofl[i][1]))
    else:
        #img here is constantly 40X(original resolution) for slideio
        start_x=int(nor_x_min*(img.rect[2]/TILE_RESO_MUL))
        start_y=int(nor_y_min*(img.rect[3]/TILE_RESO_MUL))
        end_x=int(nor_x_max*(img.rect[2]/TILE_RESO_MUL))
        end_y=int(nor_y_max*(img.rect[3]/TILE_RESO_MUL))
        desired_reso_img_width=img.rect[2]/TILE_RESO_MUL
        desired_reso_img_height=img.rect[3]/TILE_RESO_MUL
        polygon_Points=[]
        for i in range(len(nor_polygon.nor_points_lofl)):
            polygon_Points.append(Point((img.rect[2]/TILE_RESO_MUL)*nor_polygon.nor_points_lofl[i][0],(img.rect[3]/TILE_RESO_MUL)*nor_polygon.nor_points_lofl[i][1]))
        

    tile_index=0
    count_saved_tiles=0
    
    output_folder=TILES_OUTPUT_ROOT_PATH+GPname+'/'
    subfolder=''
    if this_GP_should_have_subfolder(GPname):
        subfolder=image_index_and_name+'/'
    isExist = os.path.exists(output_folder+subfolder)
    if not isExist:
        os.makedirs(output_folder+subfolder)
        print('        New directory '+output_folder+subfolder+' is created!')

    #This file should not be inside subfolder
    file_tiles_list = open(output_folder+'20230207_'+GPname+'_saved_tiles_list.txt','a+')

    if EXTRACT_TILES_OPTION==2 or EXTRACT_TILES_OPTION==3:
        file_each_image_saved_foreground_tiles_list=open(output_folder+subfolder+'20230207_'+image_index_and_name+'_'+GPname+'_saved_tiles_list.txt','a+')
    if EXTRACT_TILES_OPTION==2:
        file_portion_of_outside_annotation_polygons=open(output_folder+'20230207_'+GPname+'_portion_of_outside_annotation_polygons_saved_tiles_list.txt','a+')

    if end_x-start_x<2*tile_size and end_y-start_y<2*tile_size:
        tile_sample_step=int(tile_size/4)
    elif (end_x-start_x<3*tile_size and end_y-start_y<4*tile_size) or (end_y-start_y<3*tile_size and end_x-start_x<4*tile_size):
        tile_sample_step=int(tile_size/2)
    if tile_sample_step==0:
            tile_sample_step=1

    for x in range(start_x, end_x+1-tile_size, tile_sample_step):
        for y in range(start_y, end_y+1-tile_size, tile_sample_step):
            tile=Tile(x,y,tile_size,tile_size)
            tile_for_verifying_inside_polygon=Tile(x,y,tile_size,tile_size)
            if (end_x-start_x)*2<3*tile_size and (end_y-start_y)*2<3*tile_size:
                tile_for_verifying_inside_polygon=Tile(x+int(tile_size/4),y+int(tile_size/4),int(tile_size/2),int(tile_size/2))
            elif (end_x-start_x<4*tile_size and end_y-start_y<6*tile_size) or (end_y-start_y<4*tile_size and end_x-start_x<6*tile_size):
                tile_for_verifying_inside_polygon=Tile(x+int(tile_size/5),y+int(tile_size/5),int(3*tile_size/5),int(3*tile_size/5))
            if determine_tile_inside_polygon.tile_is_inside_polygon(polygon_Points, tile_for_verifying_inside_polygon):#This polygon could be defined to be WSI.
                if tile_is_foreground_by_sampling_dots(img, tile, GPname, image_index_and_name):
                    if EXTRACT_TILES_OPTION!=3 or tile_outside_annotation_polygons(data_json,tile, desired_reso_img_width, desired_reso_img_height):
                        if save_this_tile(GPname, tile_index):
                            if USE_OPENCV_TO_READ_IMAGE:
                                tile_image=img[y:y+tile_size, x:x+tile_size]#x is related to width, y is related to height
                            else:
                                #remember img is constantly 40X(original resolution) for slideio
                                tile_image_np=img.read_block((TILE_RESO_MUL*x,TILE_RESO_MUL*y,TILE_RESO_MUL*tile.dx,TILE_RESO_MUL*tile.dy),size=(tile.dx,tile.dy))#resize original WSI
                                tile_image = cv2.cvtColor(np.array(tile_image_np), cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_folder+subfolder+'20230207_'+image_index_and_name+'_polygon_'+str(polygon_index)+'_'+GPname+'_'+str(tile_index)+'.png',tile_image)
                            file_tiles_list.write(GPname+'/'+subfolder+'20230207_'+image_index_and_name+'_polygon_'+str(polygon_index)+'_'+GPname+'_'+str(tile_index)+'.png'+'\n')
                            if EXTRACT_TILES_OPTION==2 or EXTRACT_TILES_OPTION==3:
                                file_each_image_saved_foreground_tiles_list.write(GPname+'/'+subfolder+'20230207_'+image_index_and_name+'_polygon_'+str(polygon_index)+'_'+GPname+'_'+str(tile_index)+'.png'+'\n')
                            if EXTRACT_TILES_OPTION==2:
                                if tile_outside_annotation_polygons(data_json,tile, desired_reso_img_width, desired_reso_img_height):
                                    file_portion_of_outside_annotation_polygons.write(GPname+'/'+subfolder+'20230207_'+image_index_and_name+'_polygon_'+str(polygon_index)+'_'+GPname+'_'+str(tile_index)+'.png'+'\n')
                            count_saved_tiles+=1
                            #If we skipped tile_index==1000 because tile inside poly and OPTION==3, then we don't print
                            if tile_index%1000==0:
                                print('                Saving tile '+str(tile_index)+' x, y: '+str(x)+', '+str(y)+'  s_x:'+str(start_x)+'  e_x:'+str(end_x)\
                                    +'  s_y:'+str(start_y)+'  e_y:'+str(end_y)+'  tile_step:'+str(tile_sample_step))
                                now = datetime.now()
                                current_time = now.strftime("%H:%M:%S")
                                file_track_running_process.write('                Saving tile {}, x,y: {},{}, s_x:{}, e_x:{}, s_y:{}, e_y:{}, current time={}\n'.format(tile_index,x,y,start_x, end_x, start_y, end_y,current_time))
                            
                    tile_index+=1#We still count tile_index even if XTRACT_TILES_OPTION==3 and tile is inside polygons
            if count_saved_tiles>=tile_save_limit_in_each_polygon:
                break
        if count_saved_tiles>=tile_save_limit_in_each_polygon:
                break     
    file_tiles_list.close()
    if EXTRACT_TILES_OPTION==2 or EXTRACT_TILES_OPTION==3:
        file_each_image_saved_foreground_tiles_list.close()
    
    if EXTRACT_TILES_OPTION==2:
        file_portion_of_outside_annotation_polygons.close()

    GP_index=GP_index_in_saved_list(count_saved_tiles_precisely_in_step3, GPname)
    if GP_index<0:
        count_saved_tiles_precisely_in_step3.append([GPname,count_saved_tiles])
    else:
        count_saved_tiles_precisely_in_step3[GP_index][1]+=count_saved_tiles

    GP_index=GP_index_in_saved_list(count_saved_tiles_precisely_in_step3_in_each_image, GPname)
    if GP_index<0:
        count_saved_tiles_precisely_in_step3_in_each_image.append([GPname,count_saved_tiles])
    else:
        count_saved_tiles_precisely_in_step3_in_each_image[GP_index][1]+=count_saved_tiles

    GP_index=GP_index_in_saved_list(count_qualified_tiles_precisely_in_step3, GPname)
    if GP_index<0:
        count_qualified_tiles_precisely_in_step3.append([GPname,tile_index])
    else:
        count_qualified_tiles_precisely_in_step3[GP_index][1]+=tile_index

    GP_index=GP_index_in_saved_list(count_qualified_tiles_precisely_in_step3_in_each_image, GPname)
    if GP_index<0:
        count_qualified_tiles_precisely_in_step3_in_each_image.append([GPname,tile_index])
    else:
        count_qualified_tiles_precisely_in_step3_in_each_image[GP_index][1]+=tile_index

def save_draw_tiles_in_polygons_in_one_image(img,image_index_and_name,data_json,image_relative_path,GP_count_tiles,\
    count_tiles_roughly, count_tiles_roughly_in_one_image,count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image,\
    count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image,file_track_running_process):
    #downsample image from 40x to 20x !!
    if USE_OPENCV_TO_READ_IMAGE:
        img=cv2.resize(img, (int(img.shape[1]/TILE_RESO_MUL), int(img.shape[0]/TILE_RESO_MUL)), interpolation = cv2.INTER_AREA)
    #getting 20X in slideio takes huge time, skip that and directly handle tile, we will take care of its affect

    tile_size=SAVING_TILE_SIZE
    tile_sample_step=SAVING_TILE_JUMP_STEP

    isExist = os.path.exists(SAVE_PRESENTATION_PATH)
    if not isExist:
        os.makedirs(SAVE_PRESENTATION_PATH)
        print('New directory '+SAVE_PRESENTATION_PATH+' is created!')

    #step 1
    if ENABLE_STEP1:
        #2023: we try to constantly get 5% size of 40X WSI. 
        scale_percent = 5*TILE_RESO_MUL # percent of WSI size(WSI is resized to expected resolution), it is good option if 100/scale_percent is integer.
        if USE_OPENCV_TO_READ_IMAGE:
            resized_width = int(img.shape[1] * scale_percent / 100)
            resized_height = int(img.shape[0] * scale_percent / 100)
            resized_dim = (resized_width, resized_height)
            resized_tile_size=int(tile_size* scale_percent / 100)
            resized_tile_sample_step=int(tile_sample_step* scale_percent / 100)
            # resize image, resize will not change source image
            resized_img_for_drawing = cv2.resize(img, resized_dim, interpolation = cv2.INTER_NEAREST)
            resized_img_for_scanning = cv2.resize(img, resized_dim, interpolation = cv2.INTER_NEAREST)

        else:#slideio
            resized_width=int((img.rect[2]/TILE_RESO_MUL)* scale_percent / 100)#  '/TILE_RESO_MUL' because downsample to 20X/10X first
            resized_height=int((img.rect[3]/TILE_RESO_MUL)* scale_percent / 100)# which is 
            resized_tile_size=int(tile_size* scale_percent / 100)
            resized_tile_sample_step=int(tile_sample_step* scale_percent / 100)
            resized_img_np_for_drawing=img.read_block(size=(resized_width,resized_height))#resize original WSI
            resized_img_for_drawing = cv2.cvtColor(np.array(resized_img_np_for_drawing), cv2.COLOR_RGB2BGR)
            resized_img_np_for_scanning=img.read_block(size=(resized_width,resized_height))#resize original WSI
            resized_img_for_scanning = cv2.cvtColor(np.array(resized_img_np_for_scanning), cv2.COLOR_RGB2BGR)

        #in opencv version, img is 20X now, in slideio, img is 40X!
        step1_draw_tiles_polygons_inside_image(resized_img_for_drawing, resized_img_for_scanning, data_json, resized_tile_size, resized_tile_sample_step, image_index_and_name,\
            image_relative_path, img, scale_percent, count_tiles_roughly, count_tiles_roughly_in_one_image, file_track_running_process)
        print('        Finished step 1: drawing tiles and polygons on resized image.')
        file_track_running_process.write('Finished step 1: drawing tiles and polygons on resized image.\n')

    #Recommendation: don't use step 2. It takes time and is only useful when carefully analyzing problems. 
    ###step 2
    if ENABLE_STEP2 and USE_OPENCV_TO_READ_IMAGE:
        isExist = os.path.exists(SAVE_PRESENTATION_PATH+image_relative_path)
        if not isExist:
            os.makedirs(SAVE_PRESENTATION_PATH+image_relative_path)
            print('New directory '+SAVE_PRESENTATION_PATH+image_relative_path+' is created!')
        step2_draw_polygon_boundary_images_and_their_tiles(img, data_json, tile_size, tile_sample_step, image_index_and_name, \
        image_relative_path,GP_count_tiles)
        print('Finished step 2: saving polygon boundary images.')
    

    #step 3
    if ENABLE_STEP3:
        polygon_index=0
        if EXTRACT_TILES_OPTION==1:
            for item_json in data_json:
                geo_type=item_json['geometries']['features'][0]['geometry']['type']
                if geo_type!='Point':
                    GPname=item_json['properties']['annotations']['notes']
                    if CHANGE_SPACE_TO_UNDERLINE:
                        GPname=GPname.replace(" ", "_" )
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    if process_this_GP(GPname):
                        normalized_points_list_of_list=item_json['geometries']['features'][0]['geometry']['coordinates'][0]
                        print('\n        Saving tiles in polyon '+str(polygon_index)+': '+GPname)
                        file_track_running_process.write('\n        (In step 3)Now processing polygon {}: {}, current time={}\n'.format(polygon_index, GPname, current_time))
                        nor_polygon=Polygon(GPname,normalized_points_list_of_list)
                        #in opencv version, img is 20X now, in slideio, img is 40X!!!!!!!!!!!!!!!!!!!!!!!!!!
                        step3_save_tiles_inside_a_polygon(img, data_json, nor_polygon, tile_size, tile_sample_step, image_index_and_name, GPname, polygon_index,\
                            count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image, \
                            count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image, file_track_running_process)
                    else:
                        file_track_running_process.write('\n        (In step 3)Now skipped polygon {}: {}, current time={}\n'.format(polygon_index, GPname, current_time))
                    polygon_index+=1
        else:
            GPname=OUTSIDE_POLYGON_TYPE
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('\n        Saving tiles in : '+GPname)
            file_track_running_process.write('\n        (In step 3)Now processing {}, current time={}\n'.format(GPname, current_time))
            nor_polygon=Polygon(OUTSIDE_POLYGON_TYPE,get_nor_polygon_from_positive_region(image_index_and_name))
            step3_save_tiles_inside_a_polygon(img, data_json, nor_polygon, tile_size, tile_sample_step, image_index_and_name, GPname, polygon_index,\
                            count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image, \
                            count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image, file_track_running_process)


if __name__ == '__main__':
    if EXTRACT_TILES_OPTION==2:
        assert OUTSIDE_POLYGON_TYPE=='All_foreground_tiles'#for EXTRACT_TILES_OPTION=2: extract all foreground tiles everywhere in WSIs,   
        #1: extract foreground tiles inside polygons,  2: extract all foreground tiles everywhere in WSIs,   
        #3: extract all foreground tiles outside of all polygons. So 1 U 3 = 2
    if EXTRACT_TILES_OPTION==3:
        assert OUTSIDE_POLYGON_TYPE=='Outside_all_polygons'#for EXTRACT_TILES_OPTION=3: extract all foreground tiles outside of all polygons.

    #Question : How to Solve xlrd.biffh.XLRDError: Excel xlsx file; not supported ?
    #Answer : The latest version of xlrd(2.01) only supports .xls files. Installing the older version 1.2.0 worked for me to open .xlsx files.   
    # To open Workbook
    excel_wb = xlrd.open_workbook(excel_loc)
    excel_sheet = excel_wb.sheet_by_index(0)
    print(excel_sheet.cell_value(1, 5))

    count_image=0
    GP_dict={}#count how many polygons in total for each Growth Pattern

    GP_count_tiles={}#count in step 2. count how many tiles in total for each Growth Pattern

    count_tiles_roughly=[]#use resized image in step 1 to count, format is like [['Sarcomatoid',5],['Rhabdoid',12]]
    count_tiles_roughly_in_one_image=[]#use resized image in step 1 to count

    count_saved_tiles_precisely_in_step3=[]#saved tiles may be downsampling of qualified tiles. 
    count_saved_tiles_precisely_in_step3_in_each_image=[]

    count_qualified_tiles_precisely_in_step3=[]#saved tiles may be downsampling of qualified tiles. 
    count_qualified_tiles_precisely_in_step3_in_each_image=[]

    if INITIALIZE_LIST_WITH_SORTED_GP:
        count_tiles_roughly=initialize_GP_list()
        count_tiles_roughly_in_one_image=initialize_GP_list()
        count_saved_tiles_precisely_in_step3=initialize_GP_list()
        count_saved_tiles_precisely_in_step3_in_each_image=initialize_GP_list()
        count_qualified_tiles_precisely_in_step3=initialize_GP_list()
        count_qualified_tiles_precisely_in_step3_in_each_image=initialize_GP_list()

    isExist = os.path.exists(SAVE_PRESENTATION_PATH+'general/')
    if not isExist:
        os.makedirs(SAVE_PRESENTATION_PATH+'general/')
        print('        New directory '+SAVE_PRESENTATION_PATH+'general/'+' is created!')

    file_track_running_process=open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_track_running_process.txt','a+')

    for row in range(process_image_start,process_image_end+1):#[1,49image+1]:#16,17,29,35,36
        image_relative_path_name_ext=excel_sheet.cell_value(row, 5)[image_name_start_index_in_path:]
        json_file_name_ext=excel_sheet.cell_value(row, 6)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('{} Image is in row {} in excel file. Current time is {}'.format(row, row,current_time))
        print(image_relative_path_name_ext)
        print(json_file_name_ext)
        if not name_is_from_desired_set(image_relative_path_name_ext):
            print('Current image not in desired set(ex: training set), skip this WSI. ')
            file_track_running_process.write('\n{} Image {} in row {} not in desired set(ex: training set), skip this WSI. current time={}\n'.format(row, image_relative_path_name_ext, row, current_time))
            continue
        image_relative_path_name=image_relative_path_name_ext.split(".")
        image_relative_path_name=image_relative_path_name[:-1]
        image_relative_path_name='.'.join(image_relative_path_name)
        print('image_relative_path_name: '+image_relative_path_name)
        image_relative_path=image_relative_path_name.split("/")
        image_relative_path=image_relative_path[0]
        image_name_ext = image_relative_path_name_ext.split("/")
        image_name_ext=image_name_ext[-1]
        image_name=image_name_ext.split(".")
        image_name=image_name[:-1]
        image_name='.'.join(image_name)
        print(image_name_ext)
        print('image_name: '+image_name)
        print('Reading WSI image...')
        if USE_OPENCV_TO_READ_IMAGE:
            img=cv2.imread(ORIGINAL_IMAGES_PATH+image_relative_path_name_ext, cv2.IMREAD_UNCHANGED)#,cv2.IMREAD_UNCHANGED
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            file_track_running_process.write('\n{} Now processing image {} in row {} in Excel file, WSI shape {} x {}, current time={}\n'.format(row, image_name_ext, row, img.shape[0], img.shape[1], current_time))
            print('image shape[0] height: '+str(img.shape[0]))#height y #Generally the size is 50,000 x 60,000
            print('image shape[1] width: '+str(img.shape[1]))#width x
            print(img.shape)
        else:
            slide = slideio.open_slide(ORIGINAL_IMAGES_PATH+image_relative_path_name_ext,'SVS')
            num_scenes = slide.num_scenes
            scene = slide.get_scene(0)
            print('num_scenes: {}'.format(num_scenes))
            print(scene.name, scene.rect, scene.num_channels)#1 Image (0, 0, 100720width, 84412height) 3
            scene_width=scene.rect[2]
            scene_height=scene.rect[3]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            file_track_running_process.write('\n{} Now processing image {} in row {} in Excel file, WSI shape {} x {}, current time={}\n'.format(row, image_name_ext, row, scene_height, scene_width, current_time))
        f_json = open(ANNOTATION_FILE_PATH+json_file_name_ext,)
        data_json = json.load(f_json)
        for index in range(len(count_tiles_roughly_in_one_image)):
            count_tiles_roughly_in_one_image[index][1]=0
        for index in range(len(count_saved_tiles_precisely_in_step3_in_each_image)):
            count_saved_tiles_precisely_in_step3_in_each_image[index][1]=0
        for index in range(len(count_qualified_tiles_precisely_in_step3_in_each_image)):
            count_qualified_tiles_precisely_in_step3_in_each_image[index][1]=0
        image_relative_path=create_subfolder_name(row, image_name)
        image_index_and_name=create_subfolder_name(row, image_name)
        if USE_OPENCV_TO_READ_IMAGE:
            save_draw_tiles_in_polygons_in_one_image(img,image_index_and_name,data_json,image_relative_path,GP_count_tiles, \
                count_tiles_roughly, count_tiles_roughly_in_one_image,count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image,\
                    count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image,file_track_running_process)
        else:
            save_draw_tiles_in_polygons_in_one_image(scene,image_index_and_name,data_json,image_relative_path,GP_count_tiles, \
                count_tiles_roughly, count_tiles_roughly_in_one_image,count_saved_tiles_precisely_in_step3,count_saved_tiles_precisely_in_step3_in_each_image,\
                    count_qualified_tiles_precisely_in_step3,count_qualified_tiles_precisely_in_step3_in_each_image,file_track_running_process)
        for item_json in data_json:
            GPname=item_json['properties']['annotations']['notes']
            if CHANGE_SPACE_TO_UNDERLINE:
                GPname=GPname.replace(" ", "_" )

            if GPname in GP_dict:
                GP_dict[GPname]+=1
            else:
                GP_dict[GPname]=1

        file_count_tiles_roughly = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step1_count_tiles_roughly.txt','a+')
        file_count_tiles_roughly.write(image_index_and_name+': ')
        for GP_index in range(len(count_tiles_roughly_in_one_image)):
            file_count_tiles_roughly.write('   '+count_tiles_roughly_in_one_image[GP_index][0]+':'+str(count_tiles_roughly_in_one_image[GP_index][1]))
        file_count_tiles_roughly.write('\n')
        file_count_tiles_roughly.close()

        file_count_saved_tiles_precisely = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step3_count_saved_tiles_precisely.txt','a+')
        file_count_saved_tiles_precisely.write(image_index_and_name+': ')
        for GP_index in range(len(count_saved_tiles_precisely_in_step3_in_each_image)):
            file_count_saved_tiles_precisely.write('   '+count_saved_tiles_precisely_in_step3_in_each_image[GP_index][0]+':'+str(count_saved_tiles_precisely_in_step3_in_each_image[GP_index][1]))
        file_count_saved_tiles_precisely.write('\n')
        file_count_saved_tiles_precisely.close()

        file_count_qualified_tiles_precisely = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step3_count_qualified_tiles_precisely.txt','a+')
        file_count_qualified_tiles_precisely.write(image_index_and_name+': ')
        for GP_index in range(len(count_qualified_tiles_precisely_in_step3_in_each_image)):
            file_count_qualified_tiles_precisely.write('   '+count_qualified_tiles_precisely_in_step3_in_each_image[GP_index][0]+':'+str(count_qualified_tiles_precisely_in_step3_in_each_image[GP_index][1]))
        file_count_qualified_tiles_precisely.write('\n')
        file_count_qualified_tiles_precisely.close()

        f_json.close()
        print('\n')
        count_image+=1
        file_track_running_process.write('\n\n')
        if count_image>=count_image_limit:
            break
    file_track_running_process.write('Program End. \n\n\n\n')
    file_track_running_process.close()

    file_count_tiles_roughly = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step1_count_tiles_roughly.txt','a+')
    file_count_tiles_roughly.write('In total: image {} - {}\n'.format(process_image_start, process_image_end))
    for GP_index in range(len(count_tiles_roughly)):
        file_count_tiles_roughly.write('    '+count_tiles_roughly[GP_index][0]+' :'+str(count_tiles_roughly[GP_index][1]))
    file_count_tiles_roughly.write('\nProgram End. \n\n\n\n')
    file_count_tiles_roughly.close()

    file_count_saved_tiles_precisely = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step3_count_saved_tiles_precisely.txt','a+')
    file_count_saved_tiles_precisely.write('In total: image {} - {}. Saved tiles may be subsample of or same '.format(process_image_start, process_image_end)\
        +'as qualified tiles.(Each polygon may has max visiting tile limit. \n')
    for GP_index in range(len(count_saved_tiles_precisely_in_step3)):
        file_count_saved_tiles_precisely.write('    '+count_saved_tiles_precisely_in_step3[GP_index][0]+' :'+str(count_saved_tiles_precisely_in_step3[GP_index][1]))
    file_count_saved_tiles_precisely.write('\nProgram End. \n\n\n\n')
    file_count_saved_tiles_precisely.close()

    file_count_qualified_tiles_precisely = open(SAVE_PRESENTATION_PATH+'general/'\
        +'/20230207_step3_count_qualified_tiles_precisely.txt','a+')
    file_count_qualified_tiles_precisely.write('In total: image {} - {}. Saved tiles may be subsample of or same '.format(process_image_start, process_image_end)\
        +'as qualified tiles.(Each polygon may has max visiting tile limit. \n')
    for GP_index in range(len(count_qualified_tiles_precisely_in_step3)):
        file_count_qualified_tiles_precisely.write('    '+count_qualified_tiles_precisely_in_step3[GP_index][0]+' :'+str(count_qualified_tiles_precisely_in_step3[GP_index][1]))
    file_count_qualified_tiles_precisely.write('\nProgram End. \n\n\n\n')
    file_count_qualified_tiles_precisely.close()
    
    print('Count all polygons in scanned images.')
    for key1 in GP_dict.items():
        print(key1)
    
    file_count_polygons = open(SAVE_PRESENTATION_PATH+'general/'\
    +'/20230207_count_polygons.txt','a+')
    file_count_polygons.write('In total: image {} - {}\n'.format(process_image_start, process_image_end))
    for key1, value1 in GP_dict.items():
        file_count_polygons.write(str(key1)+': '+str(value1)+' polygons. \n')
    file_count_polygons.write('Program End. \n\n\n\n')
    file_count_polygons.close()
    print('\n')

    print('Count tiles in step2')
    for key2 in GP_count_tiles.items():
        print(key2)

    file_count_tiles = open(SAVE_PRESENTATION_PATH+'general/'\
    +'/20230207_step2_count_qualified_tiles.txt','a+')#Results should be same as step3_count_qualified_tiles.txt
    file_count_tiles.write('In total: image {} - {}\n'.format(process_image_start, process_image_end))
    for key2, value2 in GP_count_tiles.items():
        file_count_tiles.write(str(key2)+': '+str(value2)+' tiles. \n')
    file_count_tiles.write('Program End. \n\n\n\n')
    file_count_tiles.close()


    
