import os
from shutil import copyfile
import numpy as np
import random
from shutil import rmtree


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
    SAMPLE_DIR = r'C:\Users\pc\Desktop\dafenlei0809rename'  # GAI
    paths = get_all_path(SAMPLE_DIR)
    if os.path.exists(os.path.join(cur_dir, 'example/dataset')):
        rmtree(os.path.join(cur_dir, 'example/dataset'))
    os.mkdir(os.path.join(cur_dir, 'example/dataset'))
    os.mkdir(os.path.join(cur_dir, 'example/dataset/train'))
    os.mkdir(os.path.join(cur_dir, 'example/dataset/test'))


    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))
    cnt_dic = {}
    cnt_dic.update(list(zip(cats, cnt)))
    cnt = 0
    for path in paths:
        cnt += 1
        dic[path.split('\\')[-1].split('!')[0]] += 1
    print('normalized_all(', cnt, '):', dic)

    for path in paths:
        src = path
        c = path.split('\\')[-1].split('!')[0]

        if (c == '4') and (np.random.rand() < 0.25):  #太多了 用一部分就行
            continue

        if (c == '5') and (np.random.rand() < 0.1):
            continue

        if (c == '3') and (np.random.rand() < 0.1):
            continue

        if cnt_dic[c] != 200 and np.random.rand() > 0.6:
            cnt_dic[c] += 1
            dest = os.path.join(cur_dir, 'example/dataset/test', path.split('\\')[-1])
            copyfile(src, dest)
            continue

        if (c == '7') and (np.random.rand() < 0.35):
            copyfile(src, os.path.join(cur_dir, 'example/dataset/train', ''.join(path.split('\\')[-1].split('.')[:-1]) + 'x.bmp'))
            dest = os.path.join(cur_dir, 'example/dataset/train', path.split('\\')[-1])
            copyfile(src, dest)
            continue
        
        if (c == '8') and (np.random.rand() < 0.25):
            copyfile(src, os.path.join(cur_dir, 'example/dataset/train', ''.join(path.split('\\')[-1].split('.')[:-1]) + 'x.bmp'))
            dest = os.path.join(cur_dir, 'example/dataset/train', path.split('\\')[-1])
            copyfile(src, dest)
            continue

        dest = os.path.join(cur_dir, 'example/dataset/train', path.split('\\')[-1])
        copyfile(src, dest)


    SAMPLE_DIR = os.path.join(cur_dir, 'example/dataset/test')
    paths = get_all_path(SAMPLE_DIR)
    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))
    cnt = 0
    print('\nHas split from example/normalized to example/dataset/(train or test)')
    for path in paths:
        dic[path.split('\\')[-1].split('!')[0]] += 1
        cnt += 1
    print('test_all(', cnt, '):', dic)


    SAMPLE_DIR = os.path.join(cur_dir, 'example/dataset/train')
    paths = get_all_path(SAMPLE_DIR)
    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic = {}
    dic.update(list(zip(cats, cnt)))

    samples = {}
    sets = [[] for i in range(10)]
    samples.update(list(zip(cats, sets)))
    cnt = 0
    for path in paths:
        dic[path.split('\\')[-1].split('!')[0]] += 1
        samples[path.split('\\')[-1].split('!')[0]].append(path)
        cnt += 1

    traintxt = [[] for i in range(5)]
    valtxt = [[] for i in range(5)]
 
    for cat, sets in samples.items():
        random.seed(0)
        random.shuffle(sets)

        for sample_idx in range(len(sets)):
            if sample_idx % 18 == 0:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[0].append(sets[sample_idx] + '\n')
            elif sample_idx % 18 == 1:
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[1].append(sets[sample_idx] + '\n')
            elif sample_idx % 18 == 2:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[3].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[2].append(sets[sample_idx] + '\n')
            elif sample_idx % 18 == 3:
                traintxt[1].append(sets[sample_idx] + '\n')
                traintxt[2].append(sets[sample_idx] + '\n')
                traintxt[0].append(sets[sample_idx] + '\n')
                traintxt[4].append(sets[sample_idx] + '\n')
                valtxt[3].append(sets[sample_idx] + '\n')
            elif sample_idx % 18 == 4:
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
        if os.path.exists('HoleDefect\\txtdir\\' + str(fold)):
            rmtree('HoleDefect\\txtdir\\' + str(fold))
        os.mkdir('HoleDefect\\txtdir\\' + str(fold))

        file_handle = open('HoleDefect\\txtdir\\' + str(fold) + '\\train.txt', mode='a')
        file_handle.writelines(traintxt[fold])
        file_handle.close()
        file_handle = open('HoleDefect\\txtdir\\' + str(fold) + '\\val.txt', mode='a')
        file_handle.writelines(valtxt[fold])
        file_handle.close()
    print('train_all(', cnt, '):', dic)
    print('\nHas split from example/dataset//train to HoleDefect/txtdir/train or val.txt)')

    dic = {}
    list1 = [str(i) for i in range(10)]
    list2 = [0 for i in range(10)]
    dic.update(list(zip(list1, list2)))
    cnt = 0
    for i in traintxt[0]:
        cat = i.split('\\')[-1].split('!')[0]
        dic[cat] += 1
        cnt += 1
    print('train(', cnt, '):', dic)
    
    dic = {}
    list1 = [str(i) for i in range(10)]
    list2 = [0 for i in range(10)]
    dic.update(list(zip(list1, list2)))
    cnt = 0
    for i in valtxt[0]:
        cat = i.split('\\')[-1].split('!')[0]
        dic[cat] += 1
        cnt += 1
    print('val(', cnt, '):', dic)
