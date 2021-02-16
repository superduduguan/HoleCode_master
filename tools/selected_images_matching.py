import os
from shutil import copyfile


def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


selection_dir = r'C:\Users\pc\Desktop\HoleCode\selection2'
ori_img_dir1 = 'C:\\Users\\pc\\Desktop\\HoleCode\\v2.4.1\\pos\\'
ori_img_dir2 = 'C:\\Users\\pc\\Desktop\\HoleCode\\v2.4.1\\neg\\'
dest_img_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\new_label_2\\'

all_name = get_all_path(selection_dir)
for name in all_name:
    name = str(name.split('\\')[-1])
    src_name1 = ori_img_dir1 + name
    src_name2 = ori_img_dir2 + name
    try:
        copyfile(src_name1, dest_img_dir + name)
        # print(1)
    except:
        copyfile(src_name2, dest_img_dir + name)
        # print(2)
