import os
import lpips
import torch
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

# 初始化LPIPS
lpips_model = lpips.LPIPS(net='alex').cuda()  # 使用GPU进行加速

# 图像转换为Tensor
to_tensor = ToTensor()


def calculate_metrics(img1_tensor, img2_tensor):
    # 计算LPIPS
    lpips_distance = lpips_model(img1_tensor, img2_tensor).item()

    # 将Tensor转换回numpy数组并归一化到[0, 255]
    img1_np = (img1_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
    img2_np = (img2_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')

    # 计算SSIM
    ssim_value = ssim(img1_np, img2_np, multichannel=True)

    # 计算PSNR
    psnr_value = psnr(img1_np, img2_np)

    rmse_value = (((img1_np - img2_np) ** 2).mean())

    return ssim_value, psnr_value, lpips_distance, rmse_value


folder_path = 'results/maps_loss+trick100/test_latest/images'
image_files = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png'))])

# 确保图片数量是6的倍数
assert len(image_files) % 6 == 0, "The number of images in the folder must be a multiple of 6."

ssim_total3, psnr_total3, lpips_total3, rmse_total3 = 0, 0, 0,0
ssim_total2, psnr_total2, lpips_total2, rmse_total2 = 0, 0, 0,0
num_pairs = 0

for i in range(0, len(image_files), 6):
    pairs = [(image_files[i], image_files[i + 2]), (image_files[i + 1], image_files[i + 3])]

    img1_path, img2_path = pairs[0]
    img1 = to_tensor(Image.open(img1_path).convert('RGB')).unsqueeze(0).cuda()
    img2 = to_tensor(Image.open(img2_path).convert('RGB')).unsqueeze(0).cuda()
    ssim_value, psnr_value, lpips_distance,rmse_value = calculate_metrics(img1, img2)

    ssim_total3 += ssim_value
    psnr_total3 += psnr_value
    lpips_total3 += lpips_distance
    rmse_total3 += rmse_value

    img1_path2, img2_path2 = pairs[1]

    img1 = to_tensor(Image.open(img1_path2).convert('RGB')).unsqueeze(0).cuda()
    img2 = to_tensor(Image.open(img2_path2).convert('RGB')).unsqueeze(0).cuda()
    ssim_value, psnr_value, lpips_distance ,rmse_value= calculate_metrics(img1, img2)


    ssim_total2 += ssim_value
    psnr_total2 += psnr_value
    lpips_total2 += lpips_distance
    rmse_total2 += rmse_value
    num_pairs += 1


# 计算平均值
ssim_avg3 = ssim_total3 / num_pairs
psnr_avg3 = psnr_total3 / num_pairs
lpips_avg3 = lpips_total3 / num_pairs
rmse_avg3= rmse_total3 / num_pairs

ssim_avg2 = ssim_total2 / num_pairs
psnr_avg2 = psnr_total2 / num_pairs
lpips_avg2 = lpips_total2 / num_pairs
rmse_avg2= rmse_total2 / num_pairs

print(f"Average SSIM: {ssim_avg3}")
print(f"Average PSNR: {psnr_avg3}")
print(f"Average LPIPS: {lpips_avg3}")
print(f"Average MSE: {rmse_avg3}")


print(f"Average SSIM: {ssim_avg2}")
print(f"Average PSNR: {psnr_avg2}")
print(f"Average LPIPS: {lpips_avg2}")
print(f"Average MSE: {rmse_avg2}")




