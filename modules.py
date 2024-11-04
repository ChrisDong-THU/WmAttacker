import os
import glob
import random


def path_filter(path, exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    return sorted([path for path in path if path.lower().endswith(exts)])


def sample_image_path(folder_name, num):
    files = glob.glob(os.path.join(folder_name, '*.*'))
    files = path_filter(files)
    files = random.sample(files, num)
    
    return files


def add_watermark(wm_name='fusion1', wmarker_list=[], data_path=None, num=300):
    print(f'Watermarking with {wm_name}...')
    ori_path = os.path.join(data_path, 'ori_imgs/')
    ori_paths = sample_image_path(ori_path, num)
    
    output_path = os.path.join(data_path, wm_name + '/noatt')
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f'Watermarked images already exist in {output_path}')
        return
    else:
        os.makedirs(output_path, exist_ok=True)

    for i, ori_img_path in enumerate(ori_paths):
        img_name = os.path.basename(ori_img_path)
        if i < num//3:
            wmarker = wmarker_list[0]
        elif i < 2*num//3:
            wmarker = wmarker_list[1]
        else:
            wmarker = wmarker_list[2]
        
        wmarker.encode(ori_img_path, os.path.join(output_path, img_name))
        
    print(f'Finished watermarking with {wm_name}')
    

def attack_wm(att_name, wm_attacker, data_path, wm_name=None):
    print(f'Attacking images...')
    if wm_name is not None:
        wm_paths = glob.glob(os.path.join(data_path, wm_name, 'noatt/*.*'))
        wm_paths = path_filter(wm_paths)
        output_path = os.path.join(data_path, wm_name, att_name)
    else:
        wm_paths = glob.glob(os.path.join(data_path, 'ori_imgs/*.*'))
        wm_paths = path_filter(wm_paths)
        output_path = os.path.join(data_path, att_name)
    
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f'Attacked images already exist in {output_path}')
        return
    else:
        os.makedirs(output_path, exist_ok=True)
    
    output_paths = []
    for wm_path in wm_paths:
        img_name = os.path.basename(wm_path)
        output_paths.append(os.path.join(output_path, img_name))
    
    wm_attacker.attack(wm_paths, output_paths)

    print(f'Finished attacking')