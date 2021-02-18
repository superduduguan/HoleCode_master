import os
from shutil import copyfile


def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


selection_dir = r'C:\Users\pc\Desktop\HoleCode\selection3'
ori_img_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\all\\'
dest_img_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\new_label_3\\'

all_name = get_all_path(selection_dir)

for name in all_name:
    filename = str(name.split('\\')[-1]).split('!')[-1]
 
    choices = get_all_path(ori_img_dir + str(name.split('\\')[-1]).split('!')[0] + '\\' + str(name.split('\\')[-1]).split('!')[1])
    for choice in choices:


        if filename == str(choice.split('\\')[-1]):
            copyfile(choice, dest_img_dir + filename)

