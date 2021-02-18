import os
import random

train_neg_table = []
train_pos_table = []
label_dict = {}
dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\Normalized_Data\\' # Normalized_Data\\' #  只改这个
image_dir = dir + 'train'
fold_num = 5
a = []
# Add samples to the whole dataset
for filename in os.listdir(image_dir):
    samplename = filename[:-3]
    img_path = os.path.join(image_dir, filename)
    cat = filename.split('!')[1]
    # FP = 0
    if cat in ['pos']:
        # classific = 2
        train_pos_table.append(samplename)
    elif cat == 'neg':
        # classific = 0
        # if filename.split('!')[0] != 'v1.1':
        #     classific = 1
        train_neg_table.append(samplename)
    else:
        raise ValueError('Unspecified class: {}'.format(cat))

train_pos_table.sort()
train_neg_table.sort()
train_table = [[] for _ in range(fold_num)]
last = 0
for i in range(len(train_pos_table)):
    choose = i % fold_num
    train_table[choose].append(train_pos_table[i])
    last = choose + 1
for i in range(fold_num):
    print(len(train_table[i]))
for i in range(len(train_neg_table)):
    choose = (last + i) % fold_num
    train_table[choose].append(train_neg_table[i])

for i in range(fold_num):
    print(len(train_table[i]))


# random.shuffle(train_table)
for n in range(fold_num):
    test_pos = 0
    val_set = train_table[n]
    train_set = []
    for i in range(fold_num):
        if i != n:
            train_set += train_table[i]
    print(len(val_set), len(train_set))
    if not os.path.exists(str(n) + '/'):
        os.makedirs(str(n) + '/')
    with open(dir + str(n) + '\\train.txt', 'w') as f:
        for i in range(len(train_set)):
            f.write(train_set[i] + '\n')
    with open(dir + str(n) + '\\test.txt', 'w') as f:
        for i in range(len(val_set)):
            f.write(val_set[i] + '\n')
            if val_set[i].split('!')[1] == 'pos':
                test_pos += 1
    print(test_pos)
