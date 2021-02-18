import os
import random
from shutil import copyfile
# 13949
src_dir = r'C:\Users\pc\Desktop\HoleCode\all_norm'
train_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\Normalized_Data\\train\\'
test_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\Normalized_Data\\test\\'


def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


paths = get_all_path(src_dir)

random.shuffle(paths)
neg_count = 0
pos_count = 0
half_count = 0
for path in paths:
    dataset = path.split('!')[0].split('\\')[-1]
    cat = path.split('!')[1]

    train_dest = train_dir + path.split('\\')[-1]
    test_dest = test_dir + path.split('\\')[-1]
    if dataset == 'v2.4.1' and cat == 'neg':
        neg_count += 1
        if neg_count == 4:
            copyfile(path, test_dest)
            neg_count = 0
        else:
            copyfile(path, train_dest)
    elif dataset == 'v2.4.1' and cat == 'pos':
        pos_count += 1
        if pos_count == 4:
            copyfile(path, test_dest)
            pos_count = 0
        else:
            copyfile(path, train_dest)
    elif cat == 'pos' or cat == 'neg':   # TODO: 可对非v1.1的neg采样
        copyfile(path, train_dest)



