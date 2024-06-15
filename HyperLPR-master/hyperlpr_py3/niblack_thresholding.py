import cv2
import numpy as np

'''
这段代码实现了 Niblack 自适应阈值算法。主要功能包括：

1. **均值和方差计算**：使用 `cv2.boxFilter` 和 `cv2.sqrBoxFilter` 函数计算图像的局部均值和方差。
2. **阈值计算**：根据 Niblack 算法，计算阈值。阈值取决于局部均值和标准差，并使用参数 k 进行调整。
3. **二值化**：根据计算得到的阈值，对输入图像进行二值化处理，得到二值化后的图像。

这段代码可以用于图像的自适应阈值分割，常用于处理光照不均匀的图像。
'''

def niBlackThreshold(  src,  blockSize,  k,  binarizationMethod= 0 ):
    mean = cv2.boxFilter(src,cv2.CV_32F,(blockSize, blockSize),borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize), borderType = cv2.BORDER_REPLICATE)
    variance = sqmean - (mean*mean)
    stddev  = np.sqrt(variance)
    thresh = mean + stddev * float(-k)
    thresh = thresh.astype(src.dtype)
    k = (src>thresh)*255
    k = k.astype(np.uint8)
    return k


# cv2.imshow()