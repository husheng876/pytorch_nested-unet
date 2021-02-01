import os
import glob

import cv2
import numpy as np
from tqdm import tqdm


def main():
    img_size = 96

    train_path=r'inputs/data-science-bowl-2018/stage1_train/*'
    paths = glob.glob(train_path)

    print(paths)
    #创建文件
    os.makedirs(r'inputs/dsb2018_%d/images' % img_size, exist_ok=True)
    os.makedirs(r'inputs/dsb2018_%d/masks/0' % img_size, exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        #path.replace('\\','/')
        print("The path is:",path)
        file_dir= r'F:/graduateStudent/Paper/About Image/Code/medical image/pytorch-nested-unet-master'
        image_path=os.path.join(file_dir,path, 'images',os.path.basename(path) +'.png')#r'F:/graduateStudent/Paper/About Image/Code/medical image/pytorch-nested-unet-master/' \

        print(image_path)
        print("The file is exist:",os.path.exists(image_path))
        img = cv2.imread(image_path)
        print(img)
        # F:\graduateStudent\Paper\About Image\Code\medical image\pytorch-nested-unet-master\inputs\data-science-bowl-2018\stage1_train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e
        # F:\graduateStudent\Paper\About Image\Code\medical image\pytorch-nested-unet-master\inputs\data-science-bowl-2018\stage1_train\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e
        mask = np.zeros((img.shape[0], img.shape[1]))
        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            mask[mask_] = 1
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/images' % img_size,
                    os.path.basename(path) + '.png'), img)
        cv2.imwrite(os.path.join('inputs/dsb2018_%d/masks/0' % img_size,
                    os.path.basename(path) + '.png'), (mask * 255).astype('uint8'))


if __name__ == '__main__':
    main()
