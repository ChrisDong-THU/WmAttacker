import torch
import torchvision.transforms as transforms
import PIL.Image as Image


def pca_2d_ch(matrix, num_pc=512):
    matrix_mean = torch.mean(matrix)
    norm_matrix = matrix - matrix_mean
    eig_val, eig_vec = torch.linalg.eig(torch.cov(norm_matrix))
    eig_val = eig_val.real
    eig_vec = eig_vec.real
    max_num_pc = eig_val.shape[0]
    
    idx = torch.argsort(eig_val).flip(dims=[0])
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    
    assert num_pc <= max_num_pc, "num_pc should be less than or equal to the number of principal components"
    eig_vec = eig_vec[:, :num_pc]
    score = torch.matmul(eig_vec.T, norm_matrix)
    
    rec_matrix = torch.matmul(eig_vec, score) + matrix_mean
    
    return rec_matrix


def pca_2d(channels, num_pc=512):
    rec_channels = []
    for ch in range(channels.shape[0]):
        channel = channels[ch, ...]
        rec_channel = pca_2d_ch(channel, num_pc)
        rec_channels.append(rec_channel)
    
    rec_channels = torch.stack(rec_channels)
    
    return rec_channels


if __name__ == '__main__':
    image = Image.open('./workshop/ori_imgs/0.png') # h,w,c, 512x512x3
    image = transforms.ToTensor()(image)
    rec_image = pca_2d(image, 300)
    rec_image = torch.clip(rec_image, 0, 1)

    rec_image = transforms.ToPILImage()(rec_image)
    rec_image.show()
