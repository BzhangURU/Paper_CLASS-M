# This code reads Excel file and get list of all Growth Patterns appear in Excel. 
import json
import xlrd#excel
import functools

excel_loc = ("polygon_annotations_of_TCGA_WSIs/polygon_annotations_of_TCGA_WSIs/summary_of_polygon_annotations.xlsx")
process_image_start=1
process_image_end=49
CHANGE_SPACE_TO_UNDERLINE=True
ANNOTATION_FILE_PATH='polygon_annotations_of_TCGA_WSIs/polygon_annotations_of_TCGA_WSIs/polygon_annotation_details/'

def GP_index_in_saved_list(GP_list, GP_name):
    #[['SN',5],['FN',],[...]]
    for index in range(len(GP_list)):
        if GP_name==GP_list[index][0]:
            return index
    return -1

def compare_strings(x, y):#if want to sort from small to large, return x - y
    x_lower=x[0].lower()
    y_lower=y[0].lower()
    for i in range(min(len(x_lower),len(y_lower))):
        if x_lower[i]!=y_lower[i]:
            return ord(x_lower[i])-ord(y_lower[i])
    if len(x_lower)>len(y_lower):
        return 1
    else: 
        return 0
 
if __name__ == '__main__':
    excel_wb = xlrd.open_workbook(excel_loc)
    excel_sheet = excel_wb.sheet_by_index(0)
    print(excel_sheet.cell_value(1, 5))
    GP_list=[]
    geometry_dict={}

    for row in range(process_image_start,process_image_end+1):
        json_file_name_ext=excel_sheet.cell_value(row, 6)
        f_json = open(ANNOTATION_FILE_PATH+json_file_name_ext,)
        data_json = json.load(f_json)
        for item_json in data_json:
            geo_type=item_json['geometries']['features'][0]['geometry']['type']
            if geo_type!='Point':
                if geo_type not in geometry_dict:
                    geometry_dict[geo_type]=row
                GPname=item_json['properties']['annotations']['notes']
                if CHANGE_SPACE_TO_UNDERLINE:
                    GPname=GPname.replace(" ", "_" )
                #print(GPname)
                GP_index=GP_index_in_saved_list(GP_list, GPname)
                if GP_index<0:
                    GP_list.append([GPname, 1])
                else:
                    GP_list[GP_index][1]+=1
    
    GP_list_sorted=sorted(GP_list, key=functools.cmp_to_key(compare_strings))
    print(GP_list_sorted)

    for i in range(len(GP_list)):
        GP_list_sorted[i][1]=0
    print(GP_list_sorted)
    
    #for k,v in geometry_dict.items():
    print(geometry_dict.items())


#[['Alveolar', 3], ['EN', 176], ['Fat', 12], ['High_NC', 15], ['LGS', 2], ['Necrosis', 10], ['Normal', 6], ['Rhabdoid', 14], ['SN', 73], ['SSP', 8], ['Tubular', 56], ['Tubulopapillary', 18]]
#[['Alveolar', 0], ['EN', 0], ['Fat', 0], ['High_NC', 0], ['LGS', 0], ['Necrosis', 0], ['Normal', 0], ['Rhabdoid', 0], ['SN', 0], ['SSP', 0], ['Tubular', 0], ['Tubulopapillary', 0]]

