import torch
import torchvision.transforms as transforms
import PIL.Image as Image

def pca_svd(matrix, num_pc=100):
    # 将矩阵展平成二维 (C, H*W)
    C, H, W = matrix.shape
    matrix_flat = matrix.view(C, -1)
    
    # 减去每个通道的均值
    matrix_mean = torch.mean(matrix_flat, dim=1, keepdim=True)
    norm_matrix = matrix_flat - matrix_mean
    
    # 计算 SVD
    U, S, Vh = torch.linalg.svd(norm_matrix, full_matrices=False)
    
    # 选择前 num_pc 个主成分
    U_reduced = U[:, :num_pc]
    S_reduced = S[:num_pc]
    Vh_reduced = Vh[:num_pc, :]
    
    # 重建矩阵
    rec_matrix_flat = (U_reduced * S_reduced) @ Vh_reduced
    rec_matrix_flat += matrix_mean
    
    # 恢复原始形状
    rec_matrix = rec_matrix_flat.view(C, H, W)
    
    return rec_matrix
# ----------
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


def depth2space(x, block_size):
    c, h, w = x.shape[-3:]
    outer_dims = x.shape[:-3] # b
    
    s = block_size**2
    assert c%s == 0, "c should be the multiple of s"
    
    x = x.view(-1, block_size, block_size, c//s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    x = x.view(*outer_dims, c//s, h*block_size, w*block_size)
    
    return x


def space2depth(x, block_size):
    # 获取输入的形状
    c, h, w = x.shape[-3:]
    outer_dims = x.shape[:-3] # b

    s = block_size**2
    assert h%block_size == 0 and w%block_size == 0, "h and w should be the multiple of block_size"

    new_h = h // block_size
    new_w = w // block_size
    
    x = x.view(-1, c, new_h, block_size, new_w, block_size)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(*outer_dims, c * s, new_h, new_w)

    return x


def pca_3d(matrix, num_pc):
    matrix = depth2space(matrix, 2) # 1, H, W
    rec_matrix = pca_2d(matrix, num_pc)
    rec_matrix = space2depth(rec_matrix, 2)
    
    return rec_matrix


if __name__ == '__main__':
    # image = Image.open('./workshop/ori_imgs/0.png') # h,w,c, 512x512x3
    # image = transforms.ToTensor()(image)
    # rec_image = pca_2d(image, 300)
    # rec_image = torch.clip(rec_image, 0, 1)

    # rec_image = transforms.ToPILImage()(rec_image)
    # rec_image.show()
    
    t = []
    t.append(torch.zeros(1, 64, 64))
    t.append(torch.ones(1, 64, 64))
    t.append(torch.ones(1, 64, 64)*2)
    t.append(torch.ones(1, 64, 64)*3)
    t = torch.cat(t, dim=0)
    
    t = pca_3d(t, 128)
    pass
    
