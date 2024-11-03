import os
import glob


def path_filter(path, exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    return sorted([path for path in path if path.lower().endswith(exts)])


def add_watermark(wm_name, wmarker, data_path):
    print(f'Watermarking with {wm_name}...')
    ori_paths = glob.glob(os.path.join(data_path, 'ori_imgs/*.*'))
    ori_paths = path_filter(ori_paths)
    
    output_path = os.path.join(data_path, wm_name + '/noatt')
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f'Watermarked images already exist in {output_path}')
        return
    else:
        os.makedirs(output_path, exist_ok=True)

    for ori_img_path in ori_paths:
        img_name = os.path.basename(ori_img_path)
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