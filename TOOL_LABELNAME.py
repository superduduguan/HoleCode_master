import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tqdm import tqdm
from shutil import copyfile
from shutil import rmtree


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

src_fol= r'C:\Users\pc\Desktop\大分类_NORM_0806'
dst_fol = r'C:\Users\pc\Desktop\dafenlei0809rename'

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


if os.path.exists(dst_fol):
    rmtree(dst_fol)
os.mkdir(dst_fol)
for i in tqdm(get_all_path(src_fol)):
    copyfile(i, os.path.join(dst_fol, str(int(i.split('\\')[5]) - 1) + '!' + i.split('\\')[-1]))



