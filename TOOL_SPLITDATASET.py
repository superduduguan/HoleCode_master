import os
from shutil import copyfile
import numpy as np
import random


def get_all_path(input_dir):
    """
    get paths of all files in input_dir
    """
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


if __name__ == '__main__':

    np.random.seed(13)

    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    SAMPLE_DIR = os.path.join(cur_dir, 'example/normalized')
    paths = get_all_path(SAMPLE_DIR)


    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))
    cnt_dic = {}
    cnt_dic.update(list(zip(cats, cnt)))

    for path in paths:
        dic[path.split('\\')[-1].split('!')[0]] += 1
    print(dic)

    for path in paths:
        src = path
        c = path.split('\\')[-1].split('!')[0]

        if (c == '4' or c == '5') and (np.random.rand() < 0.4):
            continue

        if (c == '2') and (np.random.rand() < 0.1):
            continue

        if cnt_dic[c] != 10 and np.random.rand() > 0.8:
            cnt_dic[c] += 1
            dest = os.path.join(cur_dir, 'example/dataset/test', path.split('\\')[-1])
            copyfile(src, dest)
            continue

        if (c == '7') and (np.random.rand() < 0.15):
            copyfile(src, os.path.join(cur_dir, 'example/dataset/train', ''.join(path.split('\\')[-1].split('.')[:-1]) + 'x.bmp'))
            continue

        dest = os.path.join(cur_dir, 'example/dataset/train', path.split('\\')[-1])
        copyfile(src, dest)


    SAMPLE_DIR = os.path.join(cur_dir, 'example/dataset/test')
    paths = get_all_path(SAMPLE_DIR)
    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))

    print('Has split from normalized to dataset(train/test)')
    for path in paths:
        dic[path.split('\\')[-1].split('!')[0]] += 1
    print('test:', dic)


    SAMPLE_DIR = os.path.join(cur_dir, 'example/dataset/train')
    paths = get_all_path(SAMPLE_DIR)
    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))

    samples = {}
    sets = [[] for i in range(10)]
    samples.update(list(zip(cats, sets)))
    for path in paths:
        dic[path.split('\\')[-1].split('!')[0]] += 1
        samples[path.split('\\')[-1].split('!')[0]].append(path)

    traintxt = [[] for i in range(5)]
    valtxt = [[] for i in range(5)]
 
    for cat, sets in samples.items():
        random.seed(0)
        random.shuffle(sets)

        for sample_idx in range(len(sets)):
            if sample_idx % 8 == 0:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[0].append(sets[sample_idx] + '\n')
            elif sample_idx % 8 == 1:
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[1].append(sets[sample_idx] + '\n')
            elif sample_idx % 8 == 2:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[2].append(sets[sample_idx] + '\n')
            elif sample_idx % 8 == 3:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[3].append(sets[sample_idx] + '\n')
            elif sample_idx % 8 == 4:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
                valtxt[4].append(sets[sample_idx] + '\n')
            else:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
    
    for fold in range(5):
        if not os.path.exists('HoleDefect\\txtdir\\' + str(fold)):
            os.mkdir('HoleDefect\\txtdir\\' + str(fold))
        file_handle = open('HoleDefect\\txtdir\\' + str(fold) + '\\train.txt', mode='a')
        file_handle.writelines(traintxt[fold])
        file_handle.close()
        file_handle = open('HoleDefect\\txtdir\\' + str(fold) + '\\val.txt', mode='a')
        file_handle.writelines(valtxt[fold])
        file_handle.close()
    print('train:', dic)
    print('Has split from dataset(train) to txtdir(train/val.txt)')


