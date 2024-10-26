# -*- coding: utf-8 -*-
"""染色体图像计算机视觉处理库
本模块包含染色体图像计算机视觉处理相关的基础工具函数.
    1. 翻转变换 flip

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: May 23, 2022
"""


__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import itertools
from math import cos, fabs, radians, sin, sqrt, pi, ceil
from skimage import morphology

import cv2
import numpy as np
from matplotlib import pyplot as plt


MAX_DISTANCE = 99999
SMALL_CONTOUR_AREA_OF_CHROMO = 10
WRAPPER_SIZE_4_CHROMO_TOUCH_BORDER = 400


def feature_match_on_roi_for_flips(query_roi, target_roi):
    """通过SIFT特征匹配算法,并使用不同的翻转(flip)方式(上下翻转0, 左右翻转1, 上下左右-1),
    来计算两个ROI的相似度,
    并返回最佳的flip方式, 相似度和最佳匹配时两个图像是否颠倒

    Args:
        query_roi (_type_): Region of interest 1.
        target_roi (_type_): Region of interest 2.

    Returns:
        similarity (float): similarity score in %
        flip_idx (int): 最佳的flip方式
        upside_down (bool): 最佳匹配时两个图像是否颠倒
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Initialize BFMatcher
    bf = cv2.BFMatcher()
    # Detect keypoints and compute descriptors for target ROI
    target_kpts, target_descs = sift.detectAndCompute(target_roi, None)

    # 4个翻转方式 [不翻转, 上下翻转, 左右翻转, 上下左右翻转]
    query_flip_rois = [
        query_roi,
        cv2.flip(query_roi, 0),
        cv2.flip(query_roi, 1),
        cv2.flip(query_roi, -1),
    ]
    matches_list = []
    max_sim_flip_idx = 0
    max_sim = 0
    for roi_idx, roi in enumerate(query_flip_rois):
        # Detect keypoints and compute descriptors for query ROI
        query_kpts, query_descs = sift.detectAndCompute(roi, None)
        # Match descriptors between ROIs
        matches = bf.knnMatch(query_descs, target_descs, k=2)
        good_matches = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
        similarity = (
            len(good_matches) / max(len(query_kpts), len(target_kpts)) * 100
            if max(len(query_kpts), len(target_kpts)) > 0
            else 0
        )
        matches_list.append(
            {
                "roi_idx": roi_idx,
                "similarity": similarity,
                "good_matches": good_matches,
                "query_kpts": query_kpts,
                "query_descs": query_descs,
                "target_kpts": target_kpts,
                "target_descs": target_descs,
            }
        )
        if similarity > max_sim:
            max_sim = similarity
            max_sim_flip_idx = roi_idx

    best_flip = matches_list[max_sim_flip_idx]
    good_matches_on_best = best_flip["good_matches"]
    if len(good_matches_on_best) > 5:
        # 判断最佳匹配时两个图像是否相互颠倒
        # get matched points
        matched_pts = []
        for match in good_matches_on_best:
            query_idx = match[0].queryIdx
            train_idx = match[0].trainIdx
            query_kpt = best_flip["query_kpts"][query_idx]
            target_kpt = best_flip["target_kpts"][train_idx]
            # get matched points
            query_pt = query_kpt.pt
            target_pt = target_kpt.pt
            matched_pts.append((query_pt, target_pt))
        # 在query_roi中位于图像上半部分的匹配点,在target_roi上对应的匹配点有半数以上都位下半部分;
        # 并且，在query_roi中位于图像下半部分的匹配点,在target_roi上对应的匹配点有半数以上都位上半部分;
        # 则，说明两个图像是相互颠倒的
        pts_amt = len(good_matches_on_best)
        # print(f"pts_amt: {pts_amt}")
        # 求在query_roi中位于图像上半部分的匹配点,在target_roi上对应的匹配点位下半部分的点的个数
        query_roi_height_half = query_roi.shape[0] / 2
        target_roi_height_half = target_roi.shape[0] / 2
        mutually_upside_down_pts_amt = 0
        for matched_pt in matched_pts:
            query_pt, target_pt = matched_pt
            if query_pt[1] < query_roi_height_half and target_pt[1] > target_roi_height_half:
                mutually_upside_down_pts_amt += 1
            elif query_pt[1] > query_roi_height_half and target_pt[1] < target_roi_height_half:
                mutually_upside_down_pts_amt += 1
            else:
                continue
        mutually_upside_down = mutually_upside_down_pts_amt / pts_amt > 0.5
        upside_down = mutually_upside_down if max_sim_flip_idx in [0, 2] else not mutually_upside_down
    else:
        upside_down = False

    return max_sim, max_sim_flip_idx, upside_down


def best_feature_match_for_chromos(query_chromo, target_chromos):
    """利用特征点匹配算法计算染色体图像的相似度

    Args:
        query_chromo (numpy ndarray): 查询染色体图像
        target_chromos (list of numpy ndarray): 目标染色体图像列表

    Returns:
        最佳特征点匹配相似度, 最佳匹配的目标染色体
    """
    sim_score_max = 0
    target_chromo_on_max = None
    target_chromo_flip_idx_on_max = 0
    target_chromo_upside_down_on_max = False
    for target_chromo in target_chromos:
        try:
            sim_score, flip_idx, upside_down = feature_match_on_roi_for_flips(
                query_chromo["bbox_bbg"], target_chromo["bbox_bbg"]
            )
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            continue
        if sim_score > sim_score_max:
            sim_score_max = sim_score
            target_chromo_on_max = target_chromo
            target_chromo_flip_idx_on_max = flip_idx
            target_chromo_upside_down_on_max = upside_down
    return (
        sim_score_max,
        target_chromo_on_max,
        target_chromo_flip_idx_on_max,
        target_chromo_upside_down_on_max,
    )


def best_shape_match_for_chromos(query_chromo, target_chromos):
    """计算染色体图像的形状差异度

    Args:
        query_chromo (numpy ndarray): 查询染色体图像
        target_chromos (list of numpy ndarray): 目标染色体图像列表

    Returns:
        int: 最佳形状差异度, 最佳匹配的目标染色体
    """

    diff_score_min = 999999
    target_chromo_on_min = None
    for target_chromo in target_chromos:
        diff_score = cv2.matchShapes(query_chromo["cntr"], target_chromo["cntr"], cv2.CONTOURS_MATCH_I3, 0.0)
        if diff_score < diff_score_min:
            diff_score_min = diff_score
            target_chromo_on_min = target_chromo
    return diff_score_min * 100, target_chromo_on_min


def sift_similarity_on_roi(query_roi, target_roi):
    """通过SIFT特征匹配算法计算两个ROI的相似度, ROI是Region of Interest的缩写

    Args:
        query_roi (_type_): Region of interest 1.
        target_roi (_type_): Region of interest 2.

    Returns:
        float: similarity score in %
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for ROIs
    query_kpts, query_descs = sift.detectAndCompute(query_roi, None)
    target_kpts, target_descs = sift.detectAndCompute(target_roi, None)

    # Match descriptors between ROIs
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(query_descs, target_descs, k=2)

    good_matches = [[m] for m, n in matches if m.distance < 0.75 * n.distance]

    # 返回相似度
    return len(good_matches) / max(len(query_kpts), len(target_kpts)) * 100


def generate_distinct_colors(n):
    """获取容易区分的RGB颜色
    是在色彩空间中均匀地选择颜色，例如在HSV（色相、饱和度、亮度）空间中操作，
    然后将它们转换回RGB格式，因为HSV空间更容易生成具有不同色相的颜色。

    Args:
        n (int): 需要的颜色数

    Returns:
        (B,G,R): BGR颜色值
    """
    hues = np.linspace(0, 180, n, endpoint=False)  # HSV中的色相范围是[0, 180)
    colors = []
    for hue in hues:
        # 使用饱和度和亮度为100%来获得鲜艳的颜色
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(color.tolist())
    return colors


def cv_imread(file_path, flag=cv2.IMREAD_COLOR):
    """读取带中文路径的图片文件
    Args:
        file_path (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)  # 前值 cv2.IMREAD_COLOR


def cv_imwrite(file_path, img, ext=".png"):
    """保存带中文路径的图片文件
    Args:
        file_path (_type_): _description_
        img (_type_): _description_
    """
    cv2.imencode(ext, img)[1].tofile(file_path)


def merge_two_contours(contour1, contour2):
    """
    Merge two contours into one, from the nearest points.

    Args:
        contour1: First contour.
        contour2: Second contour.

    Returns:
        Merged contour.
    """
    (
        _,
        contour1_nearest_point_idx,
        contour2_nearest_point_idx,
    ) = get_distance_between_two_contours(contour1, contour2)

    # reorganize contour1's contour points from the nearest point
    reorganized_contour1 = np.roll(contour1, -contour1_nearest_point_idx, axis=0)
    return np.insert(contour2, contour2_nearest_point_idx, reorganized_contour1, axis=0)


def merge_two_contours_by_npi(contour1, contour2, contour1_nearest_point_idx, contour2_nearest_point_idx):
    """
    Merge two contours into one, by the nearest points index.
    "npi" means "nearest point index" of a contour.

    Args:
        contour1: First contour.
        contour2: Second contour.
        contour1_nearest_point_idx: Nearest point index of contour1.
        contour2_nearest_point_idx: Nearest point index of contour2.

    Returns:
        Merged contour.
    """
    # reorganize contour1's contour points from the nearest point
    reorganized_contour1 = np.roll(contour1, -contour1_nearest_point_idx, axis=0)
    return np.insert(contour2, contour2_nearest_point_idx, reorganized_contour1, axis=0)


def contour_closest_to_which_contour(given_contour, contours):
    """
    Find the contour in contours that is closest to the given contour.

    Args:
        contour: Given contour.
        contours: Contours to search from.

    Returns:
        Index of the closest contour in contours.
        Distance to the closest contour.
        Nearest point index of the given contour.
        Nearest point index of the closest contour.
    """
    min_distance = float("inf")
    closest_contour_idx = 0
    nearest_point_idx_on_given_contour = 0
    nearest_point_idx_on_closest_contour = 0
    for idx, cnt in enumerate(contours):
        distance, src_idx, dst_idx = get_distance_between_two_contours(given_contour, cnt)
        if distance < min_distance:
            min_distance = distance
            closest_contour_idx = idx
            nearest_point_idx_on_given_contour = src_idx
            nearest_point_idx_on_closest_contour = dst_idx

    return (
        closest_contour_idx,
        min_distance,
        nearest_point_idx_on_given_contour,
        nearest_point_idx_on_closest_contour,
    )


def get_distance_between_two_contours(contour1, contour2):
    """
    Get the distance between two contours.

    Args:
        contour1: First contour.
        contour2: Second contour.

    Returns:
        Distance between two contours.
        nearest point index of contour1.
        nearest point index of contour2.
    """
    contour1_points = contour1[:, 0, :]
    contour2_points = contour2[:, 0, :]

    min_distance = float("inf")
    contour1_nearest_point_idx = 0
    contour2_nearest_point_idx = 0

    for idx1, point1 in enumerate(contour1_points):
        for idx2, point2 in enumerate(contour2_points):
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance:
                min_distance = distance
                contour1_nearest_point_idx = idx1
                contour2_nearest_point_idx = idx2

    return min_distance, contour1_nearest_point_idx, contour2_nearest_point_idx


def contour_bbox_img(img, contour):
    """
    Get the bounding box img of a contour.

    Args:
        img: Image.
        contour: Contour.

    Returns:
        cropped: Bounding box image, with grayscale and black background.
        target_on_white: Bounding box image, with white background, not grayscale.
    """
    # if not grayscale, convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Create a mask for the contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the target
    extracted = cv2.bitwise_and(gray, gray, mask=mask)

    # get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the object with some padding, adjust padding as needed
    padding = 1
    cropped = extracted[y - padding : y + h + padding, x - padding : x + w + padding]

    # make it white background
    wbg_cropped = np.full_like(cropped, 255, dtype=np.uint8)
    np.copyto(wbg_cropped, cropped, where=(cropped > 0))

    return cropped, wbg_cropped


def erode_with_kernel(img, kernel=None, iterations=1):
    """使用指定的核进行腐蚀

    Args:
        img (numpy ndarray): 需要处理的图片
        kernel (numpy ndarray): 指定卷积核.
        iterations (int, optional): 迭代次数. Defaults to 1.

    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def dilate_with_kernel(img, kernel=None, iterations=1):
    """使用指定的核进行膨胀

    Args:
        img (numpy ndarray): 需要处理的图片
        kernel (numpy ndarray): 指定卷积核.
        iterations (int, optional): 迭代次数. Defaults to 1.

    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def gaussian_blur_with_ksize(img, ksize=5, sigma=5):
    """高斯模糊
    Args:
        img (numpy ndarray): 需要高斯模糊的图片
        ksize (int): 高斯模糊的核大小,默认为5
        sigma (int): 高斯模糊的标准差,默认为5
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur_with_ksize(img, ksize=3):
    """中值模糊
    Args:
        img (numpy ndarray): 需要高斯模糊的图片
        ksize (int): 高斯模糊的核大小,默认为5
        sigma (int): 高斯模糊的标准差,默认为5
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    return cv2.medianBlur(img, ksize)


def avg_blur_with_ksize(img, ksize=3):
    """平均模糊
    Args:
        img (numpy ndarray): 需要高斯模糊的图片
        ksize (int): 高斯模糊的核大小,默认为5
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    return cv2.blur(img, (ksize, ksize))


def gaussian_blur_with_kernel(img, kernel=None, ddepth=-1):
    """高斯模糊使用指定的核Kernel
    Args:
        img (numpy ndarray): 需要处理的图片
        kernel (numpy ndarray): 指定卷积核. 默认为None. 如果为None,则使用默认的核
        ddepth (int, optional): desired depth of the destination image, see combinations.
        Defaults to -1. when ddepth=-1, the output image will have the same depth as the source
    Returns:
        Image (numpy ndarray): 处理后的图片
    """

    if kernel is None:
        kernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32) / 9
    return cv2.filter2D(img, ddepth, kernel)


def sharpen_with_kernel(img, kernel=None, ddepth=-1):
    """用指定的核锐化图像
    Args:
        img (numpy ndarray): 需要处理的图片
        kernel (numpy ndarray): 指定卷积核. 默认为None. 如果为None,则使用默认的核
        ddepth (int, optional): desired depth of the destination image, see combinations. Defaults to -1. when ddepth=-1, the output image will have the same depth as the source
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    if kernel is None:
        kernel = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32) / 9
    return cv2.filter2D(img, ddepth, kernel)


def mean_blur_with_kernel(img, kernel=None, ddepth=-1):
    """用指定的核锐化图像
    Args:
        img (numpy ndarray): 需要处理的图片
        kernel (numpy ndarray): 指定卷积核. 默认为None. 如果为None,则使用默认的核
        ddepth (int, optional): desired depth of the destination image, see combinations. Defaults to -1. when ddepth=-1, the output image will have the same depth as the source

    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.float32) / 9
    return cv2.filter2D(img, ddepth, kernel)


def usm_sharpen(img, sigma=5):
    """USM锐化

    Args:
        img (numpy ndarray): 需要锐化的图片
        sigma (int): 高斯滤波的标准差,默认为5

    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    # USM sharpen
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (5, 5), sigma)
    return cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)


def normalization_for_ikaros_style(img, norm_alpha, norm_beta):
    """将Metafer导出的图像处理成Ikaros质素的图像

    Args:
        img (numpy ndarray): Metafer导出的,去过杂质和细胞的图像
        norm_alpha (np.uint8): 图像归一化,灰度值下限
        norm_beta (np.uint8): 图像归一化,灰度值上限

    Returns:
        Image (numpy ndarray): 处理后的贴近Ikaros风格图片
    """
    # normalize
    norm_img = cv2.normalize(img, dst=None, alpha=norm_alpha, beta=norm_beta, norm_type=cv2.NORM_MINMAX)

    odd_pixels = np.where((norm_img[:, :, 0] % 2 == 1) & (norm_img[:, :, 0] < 255))
    norm_img[odd_pixels] = norm_img[odd_pixels] + [1, 1, 1]

    # high_color_pixels1 = np.where(
    #     (norm_img[:, :, 0] > 242) & (norm_img[:, :, 0] <= 247))
    # norm_img[high_color_pixels1] = [242, 242, 242]

    # high_color_pixels = np.where((norm_img[:, :, 0] > 242) & (norm_img[:, :, 0] < 255))
    # norm_img[high_color_pixels] = [254, 254, 254]

    return norm_img


def normalization_with_contours_mask(img, contours, norm_alpha, norm_beta):
    """将对轮廓内的像素进行归一化

    Args:
        img (numpy ndarray): 需要归一化的图像
        contours (numpy ndarray): 需要归一化的轮廓
        norm_alpha (np.uint8): 图像归一化,灰度值下限
        norm_beta (np.uint8): 图像归一化,灰度值上限

    Returns:
        Image (numpy ndarray): 处理后的归一化图像
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)

    loc = np.where(mask == 255)
    values = gray_img[loc]

    norm_values = cv2.equalizeHist(values)
    norm_values = cv2.normalize(
        norm_values,
        dst=None,
        alpha=norm_alpha,
        beta=norm_beta,
        norm_type=cv2.NORM_MINMAX,
    )

    norm_img = gray_img.copy()
    for i, coord in enumerate(zip(loc[0], loc[1])):
        norm_img[coord[0], coord[1]] = norm_values[i][0]

    return cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)


def remove_img_border(img, up, down, left, right):
    """根据上下左右边界像素数删除图像边界

    Args:
        img (numpy ndarray): 要去边的图片
        up (int): 上边的宽度
        left (int): 左边的宽度
        right (int): 右边的宽度
        down (int): 下边的宽度

    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    dst_img = img.copy()
    (img_h, img_w) = dst_img.shape[:2]

    return dst_img[up : img_h - down, left : img_w - right]


def get_bkg_color(img):
    """获取图片的背景色

    Args:
        img (numpy ndarray): 需要获取背景色的图片

    Returns:
        list of BGR color value: 背景色的BGR值,比如[255, 113, 79]
    """
    height, width = img.shape[:2]
    for h, w in itertools.product(range(height), range(width)):
        color = img[h, w]
        # 第一个碰到的非黑色像素点颜色就是背景色
        if (color == np.array([0, 0, 0])).all():
            continue
        else:
            return color.tolist()


def init_canvas(width, height, color=(255, 255, 255)):
    """根据宽高和颜色初始化画布

    Args:
        width (int): 宽
        height (int): 高
        color (tuple, optional): 图像背景色. Defaults to (255, 255, 255).

    Returns:
        numpy ndarray: 初始化好的画布
    """
    canvas = np.ones((height, width, len(color)), dtype="uint8")
    canvas[:] = color
    return canvas


def init_canvas_from_shape(shape, color=(255, 255, 255)):
    """根据image shape和颜色初始化画布

    Args:
        shape (tuple): image shape
        color (tuple, optional): 图像背景色. Defaults to (255, 255, 255).

    Returns:
        numpy ndarray: 初始化好的画布
    """
    canvas = np.ones(shape, dtype="uint8")
    canvas[:] = color
    return canvas


# def show_multiple_img_in_grid(imgs, dpi=400):
#     """以表格的形式显示多张图片

#     Args:
#         imgs (list of numpy ndarray): 要显示的图片列表
#         dpi (int, optional): 要显示图片的DPI值. Defaults to 400.
#     """


def show_single_img(img, dpi=400):
    """显示单张图片

    Args:
        img (numpy ndarray): 要显示的图片
        dpi (int, optional): 要显示图片的DPI值. Defaults to 400.
    """
    plt.rcParams["figure.dpi"] = dpi
    return plt.imshow(img)


def draw_external_contours_with_idx_label(
    img,
    contours,
    contour_color=None,
    contour_thickness=1,
    contour_idx_color=None,
    contour_idx_thickness=1,
):
    """根据轮廓数据在图上把轮廓画出来,并标注出轮廓的索引

    Args:
        img (numpy ndarray): 将被画上轮廓的图片
        contours (list of numpy ndarray): 轮廓数据(包含多个轮廓数据的列表)
        contour_color (list, optional): 轮廓线颜色,缺省=为红色. Defaults to [255,0,0].
        thickness (int, optional): 轮廓线的宽度. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if contour_color is None:
        contour_color = [0, 0, 255]

    if contour_idx_color is None:
        contour_idx_color = [255, 0, 0]

    canvas = img.copy()
    cv2.drawContours(canvas, contours, -1, contour_color, contour_thickness)
    for i, cnt in enumerate(contours):
        min_area_rect = cv2.minAreaRect(cnt)
        min_area_rect = np.int64(cv2.boxPoints(min_area_rect))
        cv2.putText(
            canvas,
            f"{i}",
            min_area_rect[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            contour_idx_color,
            contour_idx_thickness,
        )

    return canvas


def draw_external_contours(img, contours, contour_color=None, thickness=1):
    """根据轮廓数据在图上把轮廓画出来

    Args:
        img (numpy ndarray): 将被画上轮廓的图片
        contours (numpy ndarray): 轮廓数据(包含多个轮廓数据的列表)
        contour_color (list, optional): 轮廓线颜色,缺省=为红色. Defaults to [255,0,0].
        thickness (int, optional): 轮廓线的宽度. Defaults to 1.

    Returns:
        _type_: _description_
    """
    if contour_color is None:
        contour_color = [255, 0, 0]
    canvas = img.copy()
    return cv2.drawContours(canvas, contours, -1, contour_color, thickness)


def find_external_contours(img, bin_thresh=None):
    """寻找图像中物体的外部轮廓

    Args:
        img (numpy ndarray): 需要找物体外部轮廓的图像
        bin_thresh (np.uint8, optional): 图像二值化阈值. Defaults to None. 为None时不做二值化操作

    Returns:
        list of contours data: 找到的轮廓数据列表
    """
    # 灰化
    dst_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if bin_thresh:
        # 二值化
        _, dst_img = cv2.threshold(dst_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    # 找轮廓
    contours, _ = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


### 找外轮廓
def find_external_contours_en(
    img,
    bin_thresh=-1,
    bin_type=cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE,
    bin_thresh_adjustment=-5,
):
    """寻找图像中物体的外部轮廓功能增强版

    Args:
        img (numpy ndarray): 需要找物体外部轮廓的图像
        bin_thresh (np.uint8, optional): 图像二值化阈值. Defaults to -1, 使用type参数指定的自适应阈值方法
        bin_type (int, optional): 二值化采用的类型. Defaults to cv2.THRESH_BINARY_INV+cv2.THRESH_TRIANGLE.
        bin_thresh_adjustment (int, optional): 使用自适应阈值时需要调整的值. Defaults to -15.

    Returns:
        _type_: _description_
    """
    # 灰化
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if bin_thresh is not None and bin_thresh >= 0:
        # 直接使用指定阈值bin_thresh进行二值化
        dst_bin_thresh, bin_img = cv2.threshold(gray_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    elif bin_thresh == -1:
        # 使用自适应阈值进行二值化
        if bin_type == cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE and bin_thresh_adjustment != 0:
            tmp_bin_thresh, _ = cv2.threshold(gray_img, 0, 255, bin_type)
            tmp_bin_thresh = tmp_bin_thresh + bin_thresh_adjustment
            if tmp_bin_thresh < 0:
                tmp_bin_thresh = 10
            dst_bin_thresh, bin_img = cv2.threshold(gray_img, tmp_bin_thresh, 255, cv2.THRESH_BINARY_INV)
        else:
            dst_bin_thresh, bin_img = cv2.threshold(gray_img, 0, 255, bin_type)
    else:
        raise ValueError("func:find_external_contours_en,bin_thresh参数错误")

    # 找轮廓
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, dst_bin_thresh
    ### END of find_external_contours_en


def find_external_contours_with_opening(img, bin_thresh=None):
    """寻找图像中物体的外部轮廓

    Args:
        img (numpy ndarray): 需要找物体外部轮廓的图像
        bin_thresh (np.uint8, optional): 图像二值化阈值. Defaults to None. 为None时不做二值化操作

    Returns:
        list of contours data: 找到的轮廓数据列表
    """
    # 灰化
    if len(img.shape) == 3:
        dst_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bin_thresh:
        # 二值化
        _, dst_img = cv2.threshold(dst_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)

    # 开运算去噪点
    kernel = np.ones((5, 5), np.uint8)
    dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # 找轮廓
    contours, _ = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contour_info_list(contours):
    """根据轮廓，获取轮廓面积,最小外接矩形,最小外接矩形四角坐标,最小外接矩形的面积,轮廓外接最小正矩形信息

    Args:
        contours (numpy ndarray): 轮廓数据列表

    Returns:
        list of dict: 轮廓信息列表,比如:[{'cnt':轮廓数据, 'area': 轮廓面积, 'min_area_rect': 轮廓最小外接矩形, 'minARarea': 轮廓最小外接矩形面积, 'min_area_rect': 最小外接矩形四角坐标, 'boundingRect': 轮廓外接最小正矩形}, ...]
    """
    cnt_info = [{} for _ in range(len(contours))]
    for idx, cnt in enumerate(contours):
        cnt_info[idx]["cnt"] = cnt

        # ((cx, cy), (width, height), theta) = cv2.minAreaRect(cnt)
        min_area_rect = cv2.minAreaRect(cnt)
        cnt_info[idx]["min_area_rect"] = min_area_rect

        min_area_rect_npint64 = np.int64(cv2.boxPoints(min_area_rect))
        cnt_info[idx]["min_area_rect_npint64"] = min_area_rect_npint64

        # (x, y, w, h) = cv2.boundingRect(cnt)
        boundingRect = np.int64(cv2.boundingRect(cnt))
        cnt_info[idx]["boundingRect"] = boundingRect

        area = cv2.contourArea(cnt)
        cnt_info[idx]["area"] = area

        minARArea = cv2.contourArea(min_area_rect_npint64)
        cnt_info[idx]["minARarea"] = minARArea
    return cnt_info


def distance(p1, p2):
    """计算两点间距离

    Args:
        p1 (tuple or list of coordinate): 其中一个点坐标
        p2 (tuple or list of coordinate): 另一个点的坐标

    Returns:
        float: 两点间的距离
    """
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    return sqrt((x**2) + (y**2))


def get_approx_min_dist_between_rect(rect1, rect2):
    """计算两个矩形间的近似最短距离,通过两个矩形的端点计算最短距离

    Args:
        rect1 (list of coordinate): 其中一个矩形的四个端点坐标,比如:[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        rect2 (list of coordinate): 另一个矩形的四个端点坐标,比如:[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        int: 两个矩形间的近似最短距离;
        [x1, y1]: 第一个矩形的最近端点坐标;
        [x2, y2]: 第二个矩形的最近端点坐标;
    """
    min_dist = MAX_DISTANCE
    nearest_pt1 = [0, 0]
    nearest_pt2 = [0, 0]
    for pt1 in rect1:
        for pt2 in rect2:
            dist = distance(pt1, pt2)
            if dist < min_dist:
                min_dist = dist
                nearest_pt1 = pt1
                nearest_pt2 = pt2
    return min_dist, nearest_pt1, nearest_pt2


def get_approx_min_dist_between_contour(contour1, contour2):
    """计算两个轮廓间的近似最短距离,通过两个轮廓最小矩形的端点计算最短距离

    Args:
        contour1 (numpy ndarray): 第一个轮廓数据
        contour2 (numpy ndarray): 第二个轮廓数据

    Returns:
        int: 两个轮廓间的近似最短距离;
        [x1, y1]: 第一个轮廓的最近端点坐标;
        [x2, y2]: 第二个轮廓的最近端点坐标;
    """
    min_ar1 = cv2.minAreaRect(contour1)
    rect1 = np.int64(cv2.boxPoints(min_ar1))
    min_ar2 = cv2.minAreaRect(contour2)
    rect2 = np.int64(cv2.boxPoints(min_ar2))

    # 遍历轮廓找到最近的点
    _, nearest_pt1, nearest_pt2 = get_approx_min_dist_between_rect(rect1, rect2)

    min_dist = MAX_DISTANCE
    for xy in contour1:
        xy = xy[0]
        dist = distance(xy, nearest_pt1)
        if dist < min_dist:
            min_dist = dist
            min_dist_pt1 = xy

    min_dist = MAX_DISTANCE
    for xy in contour2:
        xy = xy[0]
        dist = distance(xy, nearest_pt2)
        if dist < min_dist:
            min_dist = dist
            min_dist_pt2 = xy

    dist_between_contours = distance(min_dist_pt1, min_dist_pt2)

    return dist_between_contours, min_dist_pt1, min_dist_pt2


def get_min_dist_between_contour(contour1, contour2):
    """计算两个轮廓间的最短距离,并返回轮廓上距离最短的两个端点的坐标[x, y],比较耗时,慎用!!!

    Args:
        contour1 (numpy ndarray): 第一个轮廓数据
        contour2 (numpy ndarray): 第二个轮廓数据

    Returns:
        int: 两个轮廓间的最短距离
        [x1, y1]: 第一个轮廓的最近端点坐标
        [x2, y2]: 第二个轮廓的最近端点坐标
    """
    min_dist = MAX_DISTANCE
    nearest_pt1 = [0, 0]
    nearest_pt2 = [0, 0]
    for pt1 in contour1:
        for pt2 in contour2:
            pt1_xy = pt1[0]
            pt2_xy = pt2[0]
            dist = distance(pt1_xy, pt2_xy)
            if dist < min_dist:
                min_dist = dist
                nearest_pt1 = pt1_xy
                nearest_pt2 = pt2_xy
    return min_dist, nearest_pt1, nearest_pt2


def merge_contours(img_shape, contour1, contour2, nearest_pt1, nearest_pt2):
    """合并同一图像中的两个轮廓(以凸包的方式)

    Args:
        img_shape (tuple of image height, width, channel): 轮廓所在图像的shape
        contour1 (numpy ndarray): 第一个轮廓数据
        contour2 (numpy ndarray): 第二个轮廓数据
        nearest_pt1 ([x, y]): 第一个轮廓的最近端点坐标
        nearest_pt2 ([x, y]): 第二个轮廓的最近端点坐标

    Returns:
        numpy ndarray: 轮廓合并后的轮廓数据
    """
    canvas = np.zeros(img_shape, np.uint8)
    contours = [contour1, contour2]
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1)
    cv2.line(canvas, nearest_pt1, nearest_pt2, (255, 255, 255), 1)
    new_contours = find_external_contours(canvas)
    return cv2.convexHull(new_contours[0])


def contour_vertical_expansion(img_shape, contours):
    """获取垂直膨胀后的轮廓

    Args:
        img_shape (tuple of image height, width, channel): 轮廓所在图像的shape
        contour (numpy ndarray): 轮廓数据

    Returns:
        numpy ndarray: 膨胀后的轮廓数据
    """

    if len(contours) == 0:
        raise (ValueError("FUNC: contour_vertical_expansion, param: contours is empty"))

    dst_img = np.zeros(img_shape, np.uint8)
    # 变成二值图,避免mask和原图叠加后又灰色的轮廓边缘
    # 不要用THRESH_BINARY_INV,会又反一遍,轮廓变黑,背景变白
    # _, dst_img = cv2.threshold(dst_img, bin_img_thresh, 255, cv2.THRESH_BINARY)

    # 轮廓刻蚀
    cv2.drawContours(dst_img, contours, -1, (255, 255, 255), -1)

    # 垂直膨胀
    V_LINE = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    dst_img = cv2.dilate(dst_img, V_LINE, iterations=2)

    # 返回膨胀后的轮廓信息
    return find_external_contours(dst_img)


def get_chromo_img_and_mask_thru_contour_from_rep(contour, img, bgc=255):
    """把轮廓中的内容分离出来,同时提供轮廓的mask图像

    Args:
        contour (numpy ndarray): 轮廓数据
        img (numpy ndarray): 将被抠出轮廓内容的图像
        bgc (int, optional): 输出图片的背景色. Defaults to 255.

    Returns:
        numpy ndarray: 通过轮廓被抠出的图像,背景白色,图像尺寸为能被轮廓包围的最小正外接矩形(boundingRect)
        numpy ndarray: 轮廓mask图像,掩码白色,背景黑色,图像尺寸为能被轮廓包围的最小正外接矩形(boundingRect)
    """

    # 导模
    mask = np.zeros(img.shape, np.uint8)
    # mask 二值化去除 包络染色体的线框
    # _, mask = cv2.threshold(mask, bin_thresh, 255, cv2.THRESH_BINARY)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    # 取chromo
    chromo_clip = cv2.bitwise_and(img, mask)
    bBox = cv2.boundingRect(contour)
    (x, y, w, h) = bBox
    # 染色体
    chromo_clip = chromo_clip[y : y + h, x : x + w]
    # 染色体掩码
    chromo_clip_mask = mask[y : y + h, x : x + w]
    # 白背
    white_bk_chromo_clip = np.full_like(chromo_clip, bgc, dtype=np.uint8)
    np.copyto(white_bk_chromo_clip, chromo_clip, where=(chromo_clip_mask > 127))

    return white_bk_chromo_clip, chromo_clip_mask


def crop_img_from_contour(ori_img, contour):
    """根据轮廓从图中把轮廓对于物体抠出来，生成以轮廓的最小矩形方式摆正的,白底,背景透明的图像

    Args:
        ori_img (numpy ndarray): 需要被抠出轮廓内容的原图
        contour (numpy ndarray): 轮廓数据

    Returns:
        numpy ndarray: 根据轮廓抠出的,以最小矩形方式摆正的,白底,背景透明的图像
        numpy ndarray: 根据掩码图生成的轮廓数据
    """
    mask_img = np.zeros(ori_img.shape, np.uint8)
    cv2.drawContours(mask_img, [contour], 0, (255, 255, 255), -1)
    return crop_img_from_mask(ori_img, mask_img)


def crop_img_from_mask(ori_img, mask_img, bin_thresh=240):
    """掩码图罩回原图抠出图像，生成以轮廓的最小矩形方式摆正的,白底,背景透明的图像

    Args:
        ori_img (numpy ndarray): 需要被抠出轮廓内容的原图
        mask_img (numpy ndarray): 基于原图和轮廓的掩码图
        bin_thresh (int, optional): 找轮廓时的二值化阈值. Defaults to 240.

    Returns:
        numpy ndarray: 根据轮廓抠出的,以最小矩形方式摆正的,白底,背景透明的图像
        numpy ndarray: 根据掩码图生成的轮廓数据
    """

    # if not operator.eq(ori_img.shape[:2], mask_img.shape[:2]):
    #     raise(ValueError('FUNC: crop_img_from_mask, param: ori_img and mask_img shape not equal'))

    new_ori_img = ori_img.copy()
    new_mask_img = mask_img.copy()
    ori_img_bkg_color = get_bkg_color(ori_img)

    # 为了防止染色体贴边,抽正的时候被截断,先对mask和原图包边
    B = WRAPPER_SIZE_4_CHROMO_TOUCH_BORDER
    new_ori_img = cv2.copyMakeBorder(new_ori_img, B, B, B, B, cv2.BORDER_REPLICATE)
    new_mask_img = cv2.copyMakeBorder(new_mask_img, B, B, B, B, cv2.BORDER_REPLICATE)

    # 确保掩码图是二值的，否则染色体会有黑边
    _, new_mask_img = cv2.threshold(new_mask_img, bin_thresh, 255, cv2.THRESH_BINARY)

    # 通过掩码取图
    chromo_img = cv2.bitwise_and(new_ori_img, new_mask_img)

    # 白背
    white_bk_chromo_img = np.full_like(chromo_img, ori_img_bkg_color, dtype=np.uint8)
    np.copyto(white_bk_chromo_img, chromo_img, where=(new_mask_img > 0))

    # 取染色体轮廓，后面抽正用
    # 先把染色体按照正矩形割出来
    mask_contours = find_external_contours(new_mask_img)

    if len(mask_contours) == 0:
        raise (ValueError("FUNC: crop_img_from_mask, can not find contours in mask_img"))

    chromo_contour_in_ori = mask_contours[0]
    min_area_rect = cv2.minAreaRect(chromo_contour_in_ori)
    # minBox = np.int64(cv2.boxPoints(min_area_rect))

    cnt_center, cnt_size, cnt_angle = (
        min_area_rect[0],
        min_area_rect[1],
        min_area_rect[2],
    )
    cnt_center, cnt_size = tuple(map(int, cnt_center)), tuple(map(int, cnt_size))
    (cnt_w, cnt_h) = cnt_size
    if cnt_w > cnt_h:
        cnt_angle = -(90 - cnt_angle)
        cnt_w, cnt_h = cnt_h, cnt_w

    # 取得旋转矩阵
    M = cv2.getRotationMatrix2D(cnt_center, cnt_angle, 1)

    img_h, img_w = white_bk_chromo_img.shape[:2]

    # 扩大画布,新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_h = int(img_w * fabs(sin(radians(cnt_angle))) + img_h * fabs(cos(radians(cnt_angle))))
    new_w = int(img_h * fabs(sin(radians(cnt_angle))) + img_w * fabs(cos(radians(cnt_angle))))

    # 旋转阵平移
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    # 旋转
    rotated_chromo_img = cv2.warpAffine(white_bk_chromo_img, M, (new_w, new_h), borderValue=ori_img_bkg_color)
    rotated_mask_img = cv2.warpAffine(new_mask_img, M, (new_w, new_h), borderValue=[0, 0, 0])

    # 求旋转后的轮廓,这个求轮廓一定是用mask图,用chromo图在某些情况下会出两个轮廓
    # 主要原因是原图求轮廓会二值化,二值化后浅色区域会全白,造成染色体被割断
    new_contours = find_external_contours(rotated_mask_img)
    if len(new_contours) > 1:
        new_contours = [cnt for cnt in new_contours if cv2.contourArea(cnt) > SMALL_CONTOUR_AREA_OF_CHROMO]

    # 把染色体抠出来形成小图
    new_contour = new_contours[0]
    box = cv2.boundingRect(new_contour)
    (x, y, w, h) = box
    rotated_chromo_img = rotated_chromo_img[y : y + h, x : x + w]

    # 背景透明
    rotated_chromo_img = cv2.cvtColor(rotated_chromo_img, cv2.COLOR_BGR2RGBA)
    white_pixels = np.where(
        (rotated_chromo_img[:, :, 0] == 255)
        & (rotated_chromo_img[:, :, 1] == 255)
        & (rotated_chromo_img[:, :, 2] == 255)
    )
    rotated_chromo_img[white_pixels] = [0, 0, 0, 0]

    # 图片被包边,需要把轮廓数据校正回未包边的情况
    for i in range(len(chromo_contour_in_ori)):
        chromo_contour_in_ori[i][0][0] -= B
        chromo_contour_in_ori[i][0][1] -= B

    return rotated_chromo_img, chromo_contour_in_ori


def chromo_horizontal_direction_calibration(img, idx_in_pair):
    """对摆正,白底,背景透明的染色体图像,根据其在同组染色体组中的位置做水平方向左右开口的调整

    Args:
        img (numpy ndarray): 摆正,白底,背景透明的染色体图像
        idx_in_pair (_type_): 在同组染色体组中的位置,0表示最左边第一张,1表示第二张,2表示第三张,...,

    Returns:
        numpy ndarray: 开口校正后的染色体图像,idx_in_pair为0的开口向左,其他值的一律开口向右
    """
    (_, w) = img.shape[:2]
    left_half = img[:, : w // 2]
    right_half = img[:, w // 2 :]

    if (idx_in_pair == 0 and left_half.mean() < right_half.mean()) or (
        idx_in_pair != 0 and left_half.mean() > right_half.mean()
    ):
        return cv2.flip(img, 1)
    return img


def chromo_skeleton_have_crossover(chromo_contour1, chromo_contour2, img_shape):
    """判断两个染色体的骨架是否有交叉,排除仅仅是相互有粘连的情况

    Args:
        chromo_contour1 (list): 染色体轮廓
        chromo_contour2 (list): 染色体轮廓

    Returns:
        bool: 是否有交叉
    """
    sk1_img = get_skeleton_img_from_contour(chromo_contour1, img_shape)
    sk2_img = get_skeleton_img_from_contour(chromo_contour2, img_shape)
    merged_img = cv2.bitwise_or(sk1_img, sk2_img)
    return len(find_external_contours(merged_img)) == 1


def get_skeleton_img_from_contours(contours, img_shape):
    """根据轮廓提取染色体图像

    Args:
        contour (numpy ndarray): 轮廓
        img_shape (tuple): 轮廓所在图像的shape

    Returns:
        skeleton image (numpy ndarray): 轮廓的骨架图像
    """
    mask_img = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask_img, contours, -1, (255, 255, 255), -1)

    # 图像归一化
    mask_img[mask_img == 255] = 1

    # 求骨架
    skeleton_img = morphology.skeletonize(mask_img, method="lee")
    skeleton_img = skeleton_img.astype(np.uint8) * 255

    return skeleton_img


def get_skeleton_img_from_contour(contour, img_shape):
    """根据轮廓提取染色体图像

    Args:
        contour (numpy ndarray): 轮廓
        img_shape (tuple): 轮廓所在图像的shape

    Returns:
        skeleton image (numpy ndarray): 轮廓的骨架图像
    """
    mask_img = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask_img, [contour], 0, (255, 255, 255), -1)

    # 图像归一化
    mask_img[mask_img == 255] = 1

    # 求骨架
    skeleton_img = morphology.skeletonize(mask_img, method="lee")
    skeleton_img = skeleton_img.astype(np.uint8) * 255

    return skeleton_img


def chromo_stand_up_thru_mask(img, dbg=False):
    """
    :param img:传入的直立,白底,背景透明的单根染色体图像
    :return: 摆正后的染色体图片, 图片是否翻转过
    """
    (h, _, ch) = img.shape

    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if ch == 4 else img.copy()

    contours, _ = find_external_contours_en(bgr)

    mask = np.zeros(bgr.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    top_half = mask[: h // 2, :]
    bottom_half = mask[h // 2 :, :]

    top_half_mean = top_half.mean()
    bottom_half_mean = bottom_half.mean()

    if dbg:
        # 标出染色体的中间位置
        mask[h // 2, 0:2] = [0, 0, 255]
        mask[h // 2, -2:] = [0, 0, 255]
        # 标出上下半部分的均值
        cv2.putText(
            mask,
            f"{top_half_mean:.1f}",
            (2, h // 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.1,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            mask,
            f"{bottom_half_mean:.1f}",
            (2, h // 4 + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.1,
            (255, 0, 0),
            1,
        )

    if top_half_mean > bottom_half_mean:
        return (cv2.flip(img, 0), mask, True) if dbg else (cv2.flip(img, 0), True)

    return (img, mask, False) if dbg else (img, False)


def chromo_stand_up_thru_skeleton(img, dbg=False):
    """对直立,白底,背景透明的染色体图像做"抽正"操作,染色体的着丝粒在染色体的上半部分

    Args:
        img (numpy ndarray): 直立,白底,背景透明的染色体图像

    Returns:
        numpy ndarray: 抽正后的染色体图像
        Boolean: 是否进行过垂直翻转操作
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[2] == 4 else img.copy()

    # 黑背景变白背景
    bgr_wb = np.full_like(bgr, 255, dtype=np.uint8)
    np.copyto(bgr_wb, bgr, where=(bgr > 0))

    # 高斯平滑后会扩大轮廓,所以需要,扩展图片画布
    bgr_wb = enlarge_canvas(bgr_wb, 2)

    # gaussian blur
    gauss4sk = cv2.GaussianBlur(bgr_wb, (15, 15), 0)

    # 求高斯平滑后的略微平滑的边缘轮廓
    gauss4sk_cnts, _ = find_external_contours_en(gauss4sk)
    gauss4sk_cnt = gauss4sk_cnts[0]

    # smoothy gauss contour
    peri = cv2.arcLength(gauss4sk_cnt, True)
    smoothy_cnt = cv2.approxPolyDP(gauss4sk_cnt, 0.002 * peri, True)
    smoothy_sk = get_skeleton_img_from_contour(smoothy_cnt, bgr_wb.shape)

    # 求骨架上的点
    sk_h, sk_w = smoothy_sk.shape[:2]
    sk_pts = []
    for row in range(sk_h):
        sk_pts.extend([col, row] for col in range(sk_w) if smoothy_sk[row][col].any() != 0)

    # get blur for distance calculation
    # gauss4dist = cv2.GaussianBlur(bgr_wb, (3, 3), 0)
    gauss4dist = bgr_wb.copy()
    gauss4dist_cnts, _ = find_external_contours_en(gauss4dist)
    gauss4dist_cnt = gauss4dist_cnts[0]

    # 求骨架上到轮廓距离最短的点
    min_dist = 99999
    min_dist_pt = [0, 0]
    for pt in sk_pts:
        dist = cv2.pointPolygonTest(gauss4dist_cnt, pt, measureDist=True)
        if dist > 0 and dist < min_dist:
            min_dist = dist
            min_dist_pt = pt

    if dbg:
        src_with_cnt_sk_img = bgr_wb.copy()
        # 画上为求骨架模糊后的轮廓,blue
        cv2.drawContours(src_with_cnt_sk_img, [gauss4sk_cnt], 0, (255, 0, 0), 1)
        # 画上为求骨架模糊并平滑后的轮廓,red
        cv2.drawContours(src_with_cnt_sk_img, [smoothy_cnt], 0, (0, 0, 255), 1)
        # 画上为求距离而模糊后的轮廓,green
        cv2.drawContours(src_with_cnt_sk_img, [gauss4dist_cnt], 0, (0, 255, 0), 1)
        # 画上图片的高度一半的刻度线
        y = src_with_cnt_sk_img.shape[0] // 2
        src_with_cnt_sk_img[y : y + 1, 0:3] = (0, 255, 0)
        src_with_cnt_sk_img[y : y + 1, -2:] = (0, 255, 0)
        # 画上骨架
        np.copyto(src_with_cnt_sk_img, smoothy_sk, where=(smoothy_sk > 0))
        # 画出骨架上到轮廓和最短距离点
        src_with_cnt_sk_img[min_dist_pt[1], min_dist_pt[0]] = [0, 0, 255]
        src_with_cnt_sk_img[min_dist_pt[1] - 1, min_dist_pt[0]] = [0, 0, 255]
        src_with_cnt_sk_img[min_dist_pt[1] + 1, min_dist_pt[0]] = [0, 0, 255]
        src_with_cnt_sk_img[min_dist_pt[1], min_dist_pt[0] - 1] = [0, 0, 255]
        src_with_cnt_sk_img[min_dist_pt[1], min_dist_pt[0] + 1] = [0, 0, 255]

    # 判断着丝粒在上半部分还是下半部分
    h = bgr_wb.shape[0]
    if min_dist_pt[1] > h // 2:
        return (cv2.flip(img, 0), smoothy_sk, src_with_cnt_sk_img, True) if dbg else (cv2.flip(img, 0), True)

    return (img, smoothy_sk, src_with_cnt_sk_img, False) if dbg else (img, False)


def chromo_stand_up(img, dbg=False):
    return chromo_stand_up_thru_skeleton(img=img, dbg=dbg)


def _remove_metafer_label_zone(img, label_zone_h, label_zone_w):
    """去掉Metafer导出原图左上角显示图号的标签区域

    Args:
        img (numpy ndarray): Metafer导出原图

    Returns:
        numpy ndarray: 去掉左上角图号标签区域后的图像
    """
    dst_img = img.copy()
    dst_img[0:label_zone_h, 0:label_zone_w] = dst_img[label_zone_h : label_zone_h * 2, label_zone_w : label_zone_w * 2]
    return dst_img


def _chk_up_border_width(img, border_color1=None, border_color2=None):
    """检查图像上边界的宽度

    Args:
        img (numpy ndarray): 图像
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色

    Returns:
        int: 图像边界的宽度
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    (img_h, img_w) = img.shape[:2]
    up_middle_x = img_w // 2
    up_border = 0
    sequence_continue = False
    for y in range(img_h):
        if y == 0 and ((img[y][up_middle_x] == border_color1).all() or (img[y][up_middle_x] == border_color2).all()):
            up_border += 1
            sequence_continue = True
            continue

        if sequence_continue and (
            (img[y][up_middle_x] == border_color1).all() or (img[y][up_middle_x] == border_color2).all()
        ):
            up_border += 1
            sequence_continue = True
            continue

        sequence_continue = False
        break
    # 有边界为多个255和少量254组成,为了容错,边界值从index值+1,改为+3
    return up_border + 3


def _chk_right_border_width(img, border_color1=None, border_color2=None):
    """检查图像右边界的宽度

    Args:
        img (numpy ndarray): 图像
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色

    Returns:
        int: 图像边界的宽度
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    (img_h, img_w) = img.shape[:2]
    right_middle_y = img_h // 2
    right_border = 0
    sequence_continue = False
    for x in reversed(range(img_w)):
        if x == img_w - 1 and (
            (img[right_middle_y][x] == border_color1).all() or (img[right_middle_y][x] == border_color2).all()
        ):
            right_border += 1
            sequence_continue = True
            continue

        if sequence_continue and (
            (img[right_middle_y][x] == border_color1).all() or (img[right_middle_y][x] == border_color2).all()
        ):
            right_border += 1
            sequence_continue = True
            continue

        sequence_continue = False
        break
    # 有边界为多个255和少量254组成,为了容错,边界值从index值+1,改为+3
    return right_border + 3


def _chk_down_border_width(img, border_color1=None, border_color2=None):
    """检查图像下边界的宽度

    Args:
        img (numpy ndarray): 图像
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色

    Returns:
        int: 图像边界的宽度
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    (img_h, img_w) = img.shape[:2]
    down_middle_x = img_w // 2
    down_border = 0
    sequence_continue = False
    for y in reversed(range(img_h)):
        if y == img_h - 1 and (
            (img[y][down_middle_x] == border_color1).all() or (img[y][down_middle_x] == border_color2).all()
        ):
            down_border += 1
            sequence_continue = True
            continue

        if sequence_continue and (
            (img[y][down_middle_x] == border_color1).all() or (img[y][down_middle_x] == border_color2).all()
        ):
            down_border += 1
            sequence_continue = True
            continue

        sequence_continue = False
        break
    # 有边界为多个255和少量254组成,为了容错,边界值从index值+1,改为+3
    return down_border + 3


def _chk_left_border_width(img, border_color1=None, border_color2=None):
    """检查图像左边界的宽度

    Args:
        img (numpy ndarray): 图像
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色

    Returns:
        int: 图像边界的宽度
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    (img_h, img_w) = img.shape[:2]
    left_middle_y = img_h // 2
    left_border = 0
    sequence_continue = False
    for x in range(img_w):
        if x == 0 and (
            (img[left_middle_y][x] == border_color1).all() or (img[left_middle_y][x] == border_color2).all()
        ):
            left_border += 1
            sequence_continue = True
            continue

        if sequence_continue and (
            (img[left_middle_y][x] == border_color1).all() or (img[left_middle_y][x] == border_color2).all()
        ):
            left_border += 1
            sequence_continue = True
            continue

        sequence_continue = False
        break
    # 有边界为多个255和少量254组成,为了容错,边界值从index值+1,改为+3
    return left_border + 3


def remove_metafer_img_border(img, border_color1=None, border_color2=None):
    """去掉Metafer导出原图的黑白边框

    Args:
        img (numpy ndarray): Metafer导出原图
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色

    Returns:
        numpy ndarray: 去掉Metafer导出原图的黑白边框后的图像
        int: 图片顶端黑白边框的宽度
        int: 图片右边黑白边框的宽度
        int: 图片底端黑白边框的宽度
        int: 图片左边黑白边框的宽度
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    (img_h, img_w) = img.shape[:2]
    dst_img = img.copy()
    # 从四边的中心位置开始探测黑边或白边
    up_border = _chk_up_border_width(dst_img, border_color1, border_color2)
    right_border = _chk_right_border_width(dst_img, border_color1, border_color2)
    down_border = _chk_down_border_width(dst_img, border_color1, border_color2)
    left_border = _chk_left_border_width(dst_img, border_color1, border_color2)

    # 裁去边
    dst_img = dst_img[up_border : img_h - down_border, left_border : img_w - right_border]

    return dst_img


def metafer_img_clean(img, label_zone_h, label_zone_w, border_color1=None, border_color2=None):
    """去掉Metafer导出原图的黑白边框和图号标签区域

    Args:
        img (numpy ndarray): Metafer导出的原始图像,bmp,png的都行
        border_color1 (tuple): 图像边界的颜色
        border_color2 (tuple): 图像边界的颜色
    Returns:
        numpy ndarray: 清理后的图像
    """
    if border_color1 is None:
        border_color1 = [255, 255, 255]
    if border_color2 is None:
        border_color2 = [0, 0, 0]

    dst_img = img.copy()

    # 去图号区域
    dst_img = _remove_metafer_label_zone(dst_img, label_zone_h, label_zone_w)

    # 去黑白边
    dst_img = remove_metafer_img_border(dst_img, border_color1, border_color2)

    return dst_img


def _isCellLikeCircle(
    cnt,
    small_contour_area,
    circle_ratio_for_small_contour,
    circle_ratio_for_big_contour,
):
    """判断轮廓是否是细胞

    Args:
        cnt (numpy ndarray): 轮廓数据
        small_contour_area (int, optional): 小轮廓的面积阈值,小于该阈值的为小轮廓,比如21号,22号染色体. Defaults to 700.
        circle_ratio_for_small_contour (float, optional): 小轮廓的类圆率阈值,大于该阈值的轮廓被判定为非染色体. Defaults to 0.73.
        circle_ratio_for_big_contour (float, optional): 大轮廓的类圆率阈值,大于该阈值的轮廓被判定为非染色体. Defaults to 0.69.

    Returns:
        bool: 是否是细胞
    """
    area = cv2.contourArea(cnt)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    minECArea = pi * (radius**2)
    circle_ratio = area / minECArea

    return (area < small_contour_area and circle_ratio > circle_ratio_for_small_contour) or (
        area >= small_contour_area and circle_ratio > circle_ratio_for_big_contour
    )


# 判断轮廓是否触及图片边界
def _isContourTouchBorder(img_shape, contour, border_width):
    """判断轮廓是否接触图片边界

    Args:
        img (numpy ndarray): 图片
        contour (numpy ndarray): 轮廓数据
        border_width (int): 图片边界区域的宽度

    Returns:
        bool: 轮廓是否触及图片边界
    """
    x, y, w, h = cv2.boundingRect(contour)
    img_height, img_width = img_shape[:2]
    return (
        (x <= border_width)
        or (y <= border_width)
        or (x + w >= img_width - border_width)
        or (y + h >= img_height - border_width)
    )


# 判断是非染色体轮廓触及图片边界
def _isNoneChromoObjTouchBorder(img_shape, contour, border_width):
    """判断是否是非染色体的物体触及图片边界

    Args:
        img (numpy ndarray): 图片
        contour (numpy ndarray): 轮廓数据
        border_width (int): 图片边界区域的宽度

    Returns:
        bool: 是否是非染色体的物体触及图片边界
    """
    if _isContourTouchBorder(img_shape, contour, border_width):
        ((cx, cy), (width, height), theta) = cv2.minAreaRect(contour)
        if (theta < 3) or (theta > 87):
            return True
    return False


def opening(img, kernel=None, iterations=1):
    """开运算去除背景噪点

    Args:
        img (numpy ndarray): 需要去除背景噪点图片
        ksize (numpy ndarray): 开运算的核大小,为奇数3,5,7,9

    Returns:
        numpy ndarray: 经过开运算后的图片
    """
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def closing(img, kernel=None, iterations=1):
    """开运算去除背景噪点

    Args:
        img (numpy ndarray): 需要去除背景噪点图片
        ksize (numpy ndarray): 开运算的核大小,为奇数3,5,7,9

    Returns:
        numpy ndarray: 经过开运算后的图片
    """
    if kernel is None:
        kernel = np.ones((10, 10), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def patch_to_bigger_canvas(img, canvas_shape=(224, 224, 3), canvas_color=(255, 255, 255)):
    """将图片放到更大的画布上据中显示
    Args:
        img (numpy ndarray): 原图
        canvas_size (tuple, optional): 画布大小. Defaults to (224, 224).
        canvas_color (tuple, optional): 画布背景色. Defaults to (255, 255, 255).
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    img_h = img.shape[0]
    img_w = img.shape[1]

    if img_h > canvas_shape[0] or img_w > canvas_shape[1]:
        return img

    canvas = np.ones(canvas_shape, dtype=np.uint8) * canvas_color

    h_start = int((canvas_shape[0] / 2) - (img.shape[0] / 2))
    w_start = int((canvas_shape[1] / 2) - (img.shape[1] / 2))
    h_end = h_start + img_h
    w_end = w_start + img_w

    canvas[h_start:h_end, w_start:w_end] = img
    return canvas


def enlarge_canvas(img, gain=2, bgc=(255, 255, 255)):
    """放大图片的画布
    Args:
        img (numpy ndarray): 需要放大的图片
        gain(uint8, optional): 放大倍数. Defaults to 2.
        bgc, background color (tuple, optional): 背景色. Defaults to (255, 255, 255).
    Returns:
        Image (numpy ndarray): 处理后的图片
    """
    enlarged_h = ceil(img.shape[0] * gain)
    enlarged_w = ceil(img.shape[1] * gain)
    if len(img.shape) > 2:
        canvas = np.zeros((enlarged_h, enlarged_w, img.shape[2]), dtype=np.uint8)
    else:
        canvas = np.zeros((enlarged_h, enlarged_w), dtype=np.uint8)

    canvas = np.full_like(canvas, bgc, dtype=np.uint8)

    h_start = int((enlarged_h / 2) - (img.shape[0] / 2))
    w_start = int((enlarged_w / 2) - (img.shape[1] / 2))
    h_end = h_start + img.shape[0]
    w_end = w_start + img.shape[1]

    canvas[h_start:h_end, w_start:w_end] = img
    return canvas


def get_bounding_rect_for_contours(contours, img_size):
    """_summary_
    Args:
        contours (list): 存储轮廓的列表
    Returns:
        tuple: 轮廓的最小外接矩形
    """
    min_x = 99999
    min_y = 99999
    max_x = 0
    max_y = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x + w > max_x:
            max_x = x + w
        if y + h > max_y:
            max_y = y + h
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    h, w = img_size
    max_x = max_x if max_x < w else w - 1
    max_y = max_y if max_y < h else h - 1
    min_x = min_x - 2 if min_x >= 2 else min_x
    min_y = min_y - 2 if min_y - 2 >= 0 else min_y
    max_x = max_x + 3 if max_x + 3 <= w else max_x
    max_y = max_y + 3 if max_y + 3 <= h else max_y
    return min_y, max_y, min_x, max_x


def img_size_convertor(img, size):
    """将图片大小转换为指定大小

    Args:
        img (numpy ndarray): 原始图片
        size (tuple): 指定大小(h, w)

    Returns:
        numpy ndarray: 转换后的图片
    """
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = size

    dst_img = img.copy()

    if src_h > dst_h:
        up_cut_width = (src_h - dst_h) // 2
        down_cut_width = src_h - dst_h - up_cut_width
        dst_img = dst_img[up_cut_width : src_h - down_cut_width, :]
    elif src_h < dst_h:
        up_wrap_width = (dst_h - src_h) // 2
        down_wrap_width = dst_h - src_h - up_wrap_width
        dst_img = cv2.copyMakeBorder(dst_img, up_wrap_width, down_wrap_width, 0, 0, cv2.BORDER_REPLICATE)

    if src_w > dst_w:
        left_cut_width = (src_w - dst_w) // 2
        right_cut_width = src_w - dst_w - left_cut_width
        dst_img = dst_img[:, left_cut_width : src_w - right_cut_width]
    elif src_w < dst_w:
        left_wrap_width = (dst_w - src_w) // 2
        right_wrap_width = dst_w - src_w - left_wrap_width
        dst_img = cv2.copyMakeBorder(dst_img, 0, 0, left_wrap_width, right_wrap_width, cv2.BORDER_REPLICATE)

    return dst_img


def img_size_convertor_with_all_contours_kept(img, size, contours):
    """将图片大小转换为指定大小,并保留所有制定轮廓

    Args:
        img (numpy ndarray): 原始图片
        size (tuple): 指定大小(h, w)
        contours (list of numpy ndarray): 轮廓列表
    Returns:
        numpy ndarray: 转换后的图片
    """
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = size

    if src_h == dst_h and src_w == dst_w:
        return img

    if src_h <= dst_h and src_w <= dst_w:
        return patch_to_bigger_canvas(img, canvas_shape=(dst_h, dst_w, 3), canvas_color=(255, 255, 255))

    (min_y, max_y, min_x, max_x) = get_bounding_rect_for_contours(contours, img.shape[:2])
    new_img = img[min_y:max_y, min_x:max_x]
    new_img_h = max_y - min_y + 1
    new_img_w = max_x - min_x + 1

    if new_img_h == dst_h and new_img_w == dst_w:
        return new_img

    if new_img_h <= dst_h and new_img_w <= dst_w:
        return patch_to_bigger_canvas(new_img, canvas_shape=(dst_h, dst_w, 3), canvas_color=(255, 255, 255))

    if (new_img_h / dst_h) > (new_img_w / dst_w):
        new_img_w = int(new_img_w * dst_h / new_img_h)
        new_img_h = dst_h
    else:
        new_img_h = int(new_img_h * dst_w / new_img_w)
        new_img_w = dst_w

    new_img = cv2.resize(new_img, (new_img_w, new_img_h))
    return patch_to_bigger_canvas(new_img, canvas_shape=(dst_h, dst_w, 3), canvas_color=(255, 255, 255))


class Metaphaser:
    """对Metafer导出的原图进行中期图处理"""

    def __init__(
        self,
        img,
        wrapper_width=5,
        border_width=3,
        bin_thresh=236,
        open_kernel=np.ones((5, 5), np.uint8),
        open_iterations=1,
        close_kernel=np.ones((5, 5), np.uint8),
        close_iterations=1,
        gauss_ksize=(5, 5),
        tiny_contour_area=100,
        small_contour_area=700,
        circle_ratio_for_big_contour=0.69,
        circle_ratio_for_small_contour=0.73,
        norm_alpha=48,
        norm_beta=250,
        meta_img_h=1024,
        meta_img_w=1280,
        border_color1=(254, 254, 254),
        border_color2=(0, 0, 0),
        label_zone_h=0,  # 17
        label_zone_w=0,  # 38
        chromo_size_gain=1,
        sharpen_or_not=False,
        ikaros_like_or_not=False,
        img_size_norm_or_not=True,
        just_fit_all_chromo_or_not=False,
        bin_thresh_calib_param=5,  # 尽量保留染色体的随体
        contour_smoothy=False,
        contour_smoothy_param=0.002,
        normalization=True,
        dbg=False,
    ):
        """初始化

        Args:
            img (Numpy ndarray): 待处理的图片,比如Metafer导出的原图
            wrapper_width (int, optional): 包边的宽度. Defaults to 5.
            border_width (int, optional): 用于判断是否染色体或细胞接触到图片的边缘. Defaults to 3.
            bin_thresh (int, optional): 图片二值化阈值. Defaults to 236.
            open_kernel (int, optional): 开运算的核. Defaults to np.ones((5, 5), np.uint8).
            open_iterations (int, optional): 开运算的迭代次数. Defaults to 1.
            close_kernel (int, optional): 闭运算的核. Defaults to np.ones((5, 5), np.uint8).
            close_iterations (int, optional): 开运算的迭代次数. Defaults to 1.
            gauss_ksize (tuple, optional): 高斯模糊的核. Defaults to (5, 5).
            tiny_contour_area (int, optional): 微小杂质轮廓的面积阈值. Defaults to 100.
            small_contour_area (int, optional): 小轮廓的面积阈值,小于该阈值的为小轮廓,比如21号,22号染色体. Defaults to 700.
            circle_ratio_for_big_contour (float, optional): 大轮廓的类圆率阈值,大于该阈值的轮廓被判定为非染色体. Defaults to 0.69.
            circle_ratio_for_small_contour (float, optional): 小轮廓的类圆率阈值,大于该阈值的轮廓被判定为非染色体. Defaults to 0.73.
            norm_alpha (int, optional): cv2.normalize alpha parameter. Defaults to 48.
            norm_beta (int, optional): cv2.normalize beta parameter. Defaults to 250.
            meta_img_h: 原图预处理完成后系统设定的中期图的高, Defaults to 1024.(最终输出图片高度)
            meta_img_w: 原图预处理完成后系统设定的中期图的宽, Defaults to 1280,(最终输出图片宽度)
            border_color1 (tuple, optional): 图像边界的颜色1. Defaults to (255,255,255).缺省认为输入图片带黑色或白色的边框,如果图片不带边框,可以设置成对图片无效的颜色(比如：(13,131,54)),这样可以避免算法误判有边框.
            border_color2 (tuple, optional): 图像边界的颜色2. Defaults to (0,0,0).缺省认为输入图片带黑色或白色的边框,如果图片不带边框,可以设置成对图片无效的颜色(比如：(13,131,54)),这样可以避免算法误判有边框.
            label_zone_h (int, optional): Metafer原图图号标记区域的高度. Defaults to 17.如果输入图片没有图号标记区域,该参数需要设置为0.
            label_zone_w (int, optional): Metafer原图图号标记区域的宽度. Defaults to 38.如果输入图片没有图号标记区域,该参数需要设置为0.
            chromo_size_gain (float, optional): 染色体大小增益. Defaults to 1,表示不对染色体大小进行放缩.
            sharpen_or_not (bool, optional): 是否进行锐化. Defaults to False.
            ikaros_like_or_not (bool, optional): 是否进行Ikaros风格的处理. Defaults to False.
            img_size_norm_or_not (bool, optional): 是否进行图像大小的标准化. Defaults to True.
            just_fit_all_chromo_or_not (bool, optional): 是否只使用图像中所有轮廓. Defaults to False.
            bin_thresh_calib_param (int, optional): 二值化阈值TRIANGLE算法的校正值. Defaults to 5.
            contour_smoothy (boolean, optional): 是否对轮廓进行平滑处理. Defaults to False.
            contour_smoothy_param (float, optional): 轮廓平滑处理的参数. Defaults to 0.002.
            normalization (boolean, optional): 是否进行图像归一化(equalizeHist with mask & normalize). Defaults to True.
            dbg (bool, optional): 是否需要debug输出. Defaults to False.
        """
        self.img = img.copy()
        self.wrapper_width = wrapper_width
        self.border_width = border_width
        self.bin_thresh = bin_thresh
        self.open_kernel = open_kernel
        self.open_iterations = open_iterations
        self.close_kernel = close_kernel
        self.close_iterations = close_iterations
        self.gauss_ksize = gauss_ksize
        self.tiny_contour_area = tiny_contour_area
        self.small_contour_area = small_contour_area
        self.circle_ratio_for_big_contour = circle_ratio_for_big_contour
        self.circle_ratio_for_small_contour = circle_ratio_for_small_contour
        self.norm_alpha = norm_alpha
        self.norm_beta = norm_beta
        self.meta_img_h = meta_img_h
        self.meta_img_w = meta_img_w
        self.border_color1 = border_color1
        self.border_color2 = border_color2
        self.label_zone_h = label_zone_h
        self.label_zone_w = label_zone_w
        self.chromo_size_gain = chromo_size_gain
        self.sharpen_or_not = sharpen_or_not
        self.ikaros_like_or_not = ikaros_like_or_not
        self.img_size_norm_or_not = img_size_norm_or_not
        self.just_fit_all_chromo_or_not = just_fit_all_chromo_or_not
        self.bin_thresh_calib_param = bin_thresh_calib_param
        self.contour_smoothy = contour_smoothy
        self.contour_smoothy_param = contour_smoothy_param
        self.normalization = normalization
        self.dbg = dbg
        return None

    def clean_metafer_label_and_border(self):
        """去除Metafer导出图片的边框和图号区域
        Returns:
            Image (Numpy ndarray): 后的图片
        """
        clean_img = metafer_img_clean(
            self.img,
            self.label_zone_h,
            self.label_zone_w,
            self.border_color1,
            self.border_color2,
        )
        return img_size_convertor(clean_img, (self.meta_img_h, self.meta_img_w))

    def metaphase(self):
        """对图片做中期处理
            1. 去图片边框,去图号区域
            2. 求轮廓
            3. 去小轮廓,去细胞,去接触边界的大轮廓
            4. 白色背景
            5. 规定化:染色体扩大1.32倍
            6. 规定化:USM锐化
            7. 规定化:统一图片尺寸
            处理原则1: 先去掉边界,一定要在去杂质和细胞的操作完成后,再把补边以符合分辨率的要求.
        Returns:
            Numpy ndarray: 今后中期处理后的图片
        """
        # 去除图片边框
        ori_clean_img = metafer_img_clean(
            self.img,
            self.label_zone_h,
            self.label_zone_w,
            self.border_color1,
            self.border_color2,
        )

        # 求染色体轮廓
        # 高斯模糊让求出来的轮廓更加平滑
        ori_clean_gauss_img = cv2.GaussianBlur(ori_clean_img, self.gauss_ksize, 0)

        # 灰化
        gray_img = cv2.cvtColor(ori_clean_gauss_img, cv2.COLOR_BGR2GRAY)

        # 二值化
        # 获取自适应的二值化阈值
        if self.bin_thresh_calib_param == 0:
            self.bin_thresh, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
        else:
            bin_thresh, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
            # 对二值化阈值再次进行校准
            if bin_thresh - self.bin_thresh_calib_param > 0:
                self.bin_thresh = bin_thresh - self.bin_thresh_calib_param
            else:
                self.bin_thresh = bin_thresh

            _, bin_img = cv2.threshold(gray_img, self.bin_thresh, 255, cv2.THRESH_BINARY_INV)

        # 开运算,去背景噪点
        bin_open_img = opening(bin_img, kernel=self.open_kernel, iterations=self.open_iterations)

        # 闭运算,去染色体内穿孔
        bin_open_close_img = closing(bin_open_img, kernel=self.close_kernel, iterations=self.close_iterations)

        # 找轮廓
        contours = cv2.findContours(bin_open_close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # 去杂质微小轮廓
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.tiny_contour_area]

        # 去细胞
        contours = [
            cnt
            for cnt in contours
            if not _isCellLikeCircle(
                cnt,
                self.small_contour_area,
                self.circle_ratio_for_small_contour,
                self.circle_ratio_for_big_contour,
            )
        ]

        # 去接触边界的大轮廓
        contours = [
            cnt for cnt in contours if not _isNoneChromoObjTouchBorder(bin_open_close_img.shape, cnt, self.border_width)
        ]
        # contours = [
        #     cnt
        #     for cnt in contours
        #     if not _isContourTouchBorder(tmp_img.shape, cnt, self.border_width)
        # ]

        # 轮廓平滑
        if self.contour_smoothy:
            contours = [
                cv2.approxPolyDP(cnt, self.contour_smoothy_param * cv2.arcLength(cnt, True), True) for cnt in contours
            ]

        # 生成有效轮廓掩码图
        mask = np.zeros(ori_clean_img.shape, np.uint8)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

        # 白背
        dst_img = np.full_like(ori_clean_img, 255, np.uint8)
        np.copyto(dst_img, ori_clean_img, where=(mask > 127))

        # 规定化
        # 自适应直方图均衡+归一化
        if self.normalization:
            dst_img = normalization_with_contours_mask(dst_img, contours, self.norm_alpha, self.norm_beta)

        # 规定化
        # 染色体放大
        if self.chromo_size_gain != 1 and self.chromo_size_gain > 0 and self.chromo_size_gain < 4:
            (h, w) = dst_img.shape[:2]
            gain = self.chromo_size_gain
            dst_img = cv2.resize(dst_img, (round(w * gain), round(h * gain)))

        # img just fit for all chromosomes
        if self.just_fit_all_chromo_or_not:
            (min_y, max_y, min_x, max_x) = get_bounding_rect_for_contours(contours, dst_img.shape[:2])
            dst_img = dst_img[min_y:max_y, min_x:max_x]
        elif self.img_size_norm_or_not:
            # 规定化
            # 变换图片尺寸
            dst_img = img_size_convertor_with_all_contours_kept(dst_img, (self.meta_img_h, self.meta_img_w), contours)

        if self.dbg:
            cv2.drawContours(dst_img, contours, -1, (255, 0, 0), 1)

        return dst_img
