import cv2
import numpy as np 


# 2d matrix pca
def pca_2d_ch(matrix, num_pc=512):
    matrix_mean = np.mean(matrix, axis=1)
    norm_matrix = matrix - matrix_mean
    eig_val, eig_vec = np.linalg.eig(np.cov(norm_matrix))
    max_num_pc = eig_val.shape[0]
    
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    
    assert num_pc <= max_num_pc, "num_pc should be less than or equal to the number of principal components"
    eig_vec = eig_vec[:, :num_pc]
    score = np.dot(eig_vec.T, norm_matrix)
    
    rec_matrix = np.dot(eig_vec, score) + matrix_mean.T
    
    return rec_matrix


def pca_2d(channels, num_pc=512):
    rec_channels = []
    for ch in range(channels.shape[-1]):
        channel = channels[:, :, ch]
        rec_channel = pca_2d_ch(channel, num_pc)
        rec_channels.append(rec_channel)
    
    rec_channels = np.stack(rec_channels, axis=-1)
    return rec_channels


image = cv2.imread('./workshop/ori_imgs/0.png') # h,w,c, 512x512x3
rec_image = pca_2d(image, 300)
rec_image = np.clip(rec_image, 0, 255).astype(np.uint8)

cv2.imshow('rec_image', rec_image)
cv2.waitKey(0)
