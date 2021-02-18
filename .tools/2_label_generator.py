import os
import json


def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


input_dir = r'C:\Users\pc\Desktop\HoleCode\json3\outputs'
output_dir = 'C:\\Users\\pc\\Desktop\\HoleCode\\json3\\clean_outputs\\'
all_paths = get_all_path(input_dir)
for path in all_paths:
    # path = r'C:\Users\pc\Desktop\HoleCode\new_label\clean_outputs\1.json'
    try:
        f = open(path, 'r')
        a = json.load(f)
        path = a['path']
        name = str(path.split('\\')[-1])
        path = str(path.split('\\')[-2] + '\\' + path.split('\\')[-1])
        print(path)

        outputs = a['outputs']
        bndbox = outputs['object'][-1]['bndbox']
        bndbox = [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax']]
        [xmin, ymin, xmax, ymax] = bndbox
        xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
        radius = (xmax - xmin) / 4 + (ymax - ymin) / 4
        dic = {'path': path, 'class': 1, 'bndbox': bndbox, 'center': [xcenter, ycenter], 'radius': radius}
        print(dic)
        with open(output_dir+name[:-4]+'.json', "w") as f:
            json.dump(dic, f)
    except Exception as e:
        print('!!!!!!!!!', path)
        quit()






# {"path": "v1.1/neg/20200723/p42b05320-114/01/neg_p42b05320-114_01_a_a0_000009.bmp", "class": 1, "bndbox": [31, 31, 71, 70], "center": [51.0, 50.5], "radius": 19.75}