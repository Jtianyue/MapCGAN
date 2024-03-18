import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def clear_image(clear_img_path, hazy_img_path):
    """
    该函数用于读取图像并计算去雾前后图像的PSNR与SSIM值。
    :param clear_img_path: 清晰图像路径
    :param hazy_img_path: 待去雾图像路径
    :return: None
    """
    # 读取清晰图像和待去雾图像
    real = cv2.imread(clear_img_path)
    hazy_img = cv2.imread(hazy_img_path)

    # 检查图像尺寸是否一致，若不一致则进行缩放
    if clear_img.shape[0] != hazy_img.shape[0] or clear_img.shape[1] != hazy_img.shape[1]:
        pil_img = Image.fromarray(hazy_img)
        pil_img = pil_img.resize((clear_img.shape[1], clear_img.shape[0]))
        hazy_img = np.array(pil_img)

    # 计算PSNR和SSIM值，PSNR越大表示图像质量越好，SSIM越大表示两图像越相似
    PSNR = peak_signal_noise_ratio(clear_img, hazy_img)
    print('PSNR: ', PSNR)
    SSIM = structural_similarity(clear_img, hazy_img, channel_axis=2)
    print('SSIM: ', SSIM)


# 测试函数
clear_img_path = "/path/to/clear/1398.png"
hazy_img_path = "/path/to/hazy/1398_5.png"
clear_image(clear_img_path, hazy_img_path)
