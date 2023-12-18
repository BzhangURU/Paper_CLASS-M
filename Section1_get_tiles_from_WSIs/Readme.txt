Section 1 is used to generate tiles for training (labeled + unlabeled), validation and test. 
The section is introduced in our CLASS-M paper https://arxiv.org/abs/2312.06978
Please contact authors to directly get tiles without performing Section 1. 

We have assigned indexes to the Whole Slide Images (WSIs).
Index 271-300 are treated as labeled training WSIs with polygon annotations.
Index 1-270 are treated as unlabeled training WSIs.
Index 301-360 are treated as validation WSIs with polygon annotations.
Index 361-420 are treated as test WSIs with polygon annotations. 
Index info can be found at polygon_annotations_of_WSIs/20230205_summary_of_polygon_annotations.xlsx
The labeled training tiles are generated inside polygons in Index 271-300.
The unlabeled training tiles are generated outside polygons in Index 271-300 and all areas in Index 001-270.
The validation tiles are generated inside polygons in Index 301-360.
The test tiles are generated inside polygons in Index Index 361-420.

We used 420 WSIs in training, validation and test.
The list of 420 WSIs can be found at 
polygon_annotations_of_WSIs/20230205_summary_of_polygon_annotations.xlsx
Please download those TCGA ccRCC Whole Slide Images (WSIs) into TCGA_ccRCC_WSIs folder.
The link to download is https://portal.gdc.cancer.gov/, please search WSI name for downloading.

The annotations of the 420 WSIS are already inside polygon_annotations_of_WSIs folder. 

To run the program Crop_tiles_based_on_polygons_in_WSIs.py
1. Set input and output paths. (Already set in the code)
Inputs:
ORIGINAL_IMAGES_PATH is the path for TCGA ccRCC WSIs.
excel_loc is the path for Excel file that saves summary of all WSIs annotations.
ANNOTATION_FILE_PATH is the path that contains details for each WSI's polygon annotation. 

Outputs:
SAVE_PRESENTATION_PATH is the path that saves low-resolution visualization of WSIs and annotated polygons.
TILES_OUTPUT_ROOT_PATH is the path that saves cropped tiles inside WSIs. 

2. Get tiles inside polygons from index 271-300 to form labeled training samples. 
Set process_image_start=271, process_image_end=300, EXTRACT_TILES_OPTION=1, OUTSIDE_POLYGON_TYPE='All_foreground_tiles'
Then run the code. 

3. Get (tiles outside polygons from index 271-300) and (tiles in all foreground areas from index 1-270) to form unlabeled training samples. 
Set process_image_start=271, process_image_end=300, EXTRACT_TILES_OPTION=3, OUTSIDE_POLYGON_TYPE='Outside_all_polygons'
Run the code. 
Then set process_image_start=1, process_image_end=270, EXTRACT_TILES_OPTION=2, OUTSIDE_POLYGON_TYPE='All_foreground_tiles'
Run the code again. 

4. Get tiles inside polygons from index 301-360 to form validation samples. 
Set process_image_start=301, process_image_end=360, EXTRACT_TILES_OPTION=1, OUTSIDE_POLYGON_TYPE='All_foreground_tiles'
Then run the code. 

5. Get tiles inside polygons from index 361-420 to form validation samples. 
Set process_image_start=361, process_image_end=420, EXTRACT_TILES_OPTION=1, OUTSIDE_POLYGON_TYPE='All_foreground_tiles'
Then run the code. 

We have provided examples of generated visualization of WSIs with annotated polygons and tiles in folder several_examples_after_running_the_code

We further checked the tiles. Unqualified tiles are marked in grey boxes in 
polygon_annotations_of_TCGA_WSIs/polygon_annotation_details/verification_after_cropping_larger_tiles.xlsx
Those tiles are not used in CLASS-M model training, validation and test. 

