import cv2
import numpy as np

from utils.chromo_cv_utils import find_external_contours

# read image
ori_img = cv2.imread('./inputs/cavt-ca-mask-eval/ori-2024_10_16_15_18_26_670f684fc4ba273a0afb83a2_00001.jpg')
msk_img = cv2.imread('./inputs/cavt-ca-mask-eval/msk-2024_10_16_15_18_26_670f684fc4ba273a0afb83a2_00001a.png')

gry_msk_img = cv2.cvtColor(msk_img, cv2.COLOR_BGR2GRAY) if len(msk_img.shape) == 3 else msk_img

# save gry_msk_img
cv2.imwrite('./inputs/cavt-ca-mask-eval/cavt-ca-mask-eval-gry.png', gry_msk_img)

msk_f = gry_msk_img > 1

wbg = np.ones_like(ori_img) * 255

wbg[msk_f] = ori_img[msk_f]

# save
cv2.imwrite('./inputs/cavt-ca-mask-eval/cavt-ca-mask-eval.png', wbg)