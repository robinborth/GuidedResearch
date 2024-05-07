import cv2


def biliteral_filter(image, dilation, sigma_color, sigma_space):
    return cv2.bilateralFilter(
        src=image,
        d=dilation,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
