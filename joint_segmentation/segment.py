import matplotlib.pyplot as pyplot
import cv2
import numpy
import numpy as np
import skimage.color
from skimage import segmentation

# constants
SAT_MAX = 40

def test():
    """Test function to show image"""
    test_img = pyplot.imread('test_data/image (2).png')
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    pyplot.imshow(gray_img)
    pyplot.show()


def split_mask(mask):
    """
    Assigns corresponding region number to edge pixels for each solder region

    :param mask:
    :return:
    """
    result = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    return_img = np.zeros(mask.shape)

    counter = 0
    for region in result:
        pixels, blank1, blank2 = region.shape
        if pixels > 25:
            for pixel in region:
                return_img[pixel[0][1]][pixel[0][0]] = counter
            counter += 1

    return return_img

def square_mask(mask, overscan, min_size):
    """
    Find square regions corresponding to contiguous segments within the image masks

    The mask should describe the solder joint regions of the image
    """
    result = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    crop_list = []
    rows, cols = mask.shape

    for region in result:
        pixels, blank1, blank2 = region.shape
        if pixels > min_size:
            crop_elt = np.zeros(shape=(4), dtype=int)

            overscan_r = (overscan / 100) * (int(max(region[:, :, 1])) - int(min(region[:, :, 1])))
            overscan_c = (overscan / 100) * (int(max(region[:, :, 0])) - int(min(region[:, :, 0])))

            crop_elt[0] = max(int(min(region[:, :, 1])) - overscan_r, 0)
            crop_elt[1] = min(int(max(region[:, :, 1])) + overscan_r, rows)
            crop_elt[2] = max(int(min(region[:, :, 0])) - overscan_c, 0)
            crop_elt[3] = min(int(max(region[:, :, 0])) + overscan_c, cols)

            crop_list.append(crop_elt)

    return crop_list

def build_segment_array(image, seg_list):
    """
    Build an array of image segments for use by classification algorithm

    These will always be square, and cover region to be segmented
    """
    seg_array = []

    for elt in seg_list:
        segment = image[elt[0]:elt[1], elt[2]:elt[3], :]
        seg_array.append(segment)

    return seg_array

def gen_solder_mask(image, gray_floor=145, sat_max=40):
    """
    Generates solder mask image

    :param image:
    :return:
    """

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light_gray = (0, 0, gray_floor)
    dark_gray = (255, sat_max, 240)
    mask = cv2.inRange(image_hsv, light_gray, dark_gray)

    # remove stray pixels
    mask = cv2.medianBlur(mask, 7)

    return mask

def auto_split_gray(image):
    """
    Find the gray value that segments the solder joints

    Real images are distributed so that there is a threshold where the number
    included begins to increase rapidly, from picking up random parts. Need
    to find this gray-floor threshold

    :returns (int) Threshold to segment the image
    """

    # find change in pixels for given gray floor
    mask_arr = []
    for gray_thr in range(0, 255):
        mask = gen_solder_mask(image, gray_floor=gray_thr, sat_max=SAT_MAX)
        mask_arr.append(mask)
    y_values = np.zeros(shape=(254), dtype=int)
    for i in range(0, 254):
        y_values[i] = int(sum(sum(mask_arr[i]/255))) - int(sum(sum(mask_arr[i+1]/255)))

    diff_y_values = np.diff(y_values)
    filtered_values = np.convolve(diff_y_values, (np.ones(shape=9)/9), 'valid')

    # find point just past peak, capture maximum solder joint area
    max_val = max(max(filtered_values), abs(min(filtered_values)))
    key_index = len(filtered_values) - 15
    while(abs(filtered_values[key_index]) < 0.5 * max_val):
        key_index -= 1
    key_index += 15

    return key_index

def resize(img, bw=False):
    """ Convert images to standard size"""
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rows, cols, colors = img.shape
    dim = max(rows, cols)

    new_image = np.zeros((dim, dim, 3), dtype=np.uint8)

    # determine upper-left corner
    r_off = int((dim - rows)/2)
    c_off = int((dim - cols)/2)

    new_image[r_off:r_off+rows,c_off:c_off+cols,:] = img

    if bw:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    return skimage.transform.resize(new_image, (150,150))

def demo():
    """Driver program to perform segmentation only"""
    test_img = cv2.imread('test_data/image (1).png')
    #gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.GaussianBlur(gray_img, (7,7), 0)
    #pyplot.imshow(gray_img, cmap="gray")
    #pyplot.show()

    # histogram, blank = np.histogram(gray_img, bins=256)
    #
    #
    # thresh, image = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # pyplot.imshow(image, cmap="gray")
    # pyplot.show()
    """ Sep old code"""
    # super_test_img = segmentation.slic(test_img, n_segments=2000, compactness=0.1, start_label=0)
    # vis_image = skimage.color.label2rgb(super_test_img, test_img)
    # pyplot.imshow(vis_image)
    #
    # cut = skimage.graph.rag_mean_color(test_img, super_test_img, mode='similarity')
    # cut_applied = skimage.graph.cut_normalized(super_test_img, cut)
    # pyplot.imshow(skimage.color.label2rgb(cut_applied, test_img))

    """Sep segmentation"""
    test_img = cv2.GaussianBlur(test_img, (5,5), 0)
    mask = gen_solder_mask(test_img, gray_floor=auto_split_gray(test_img))
    pyplot.imshow(mask, cmap="gray")

    res_img = test_img.shape
    res_img = cv2.bitwise_or(test_img, res_img, mask=mask)
    #pyplot.imshow(res_img, cmap='hsv')

    # need to find the centers of the objects
    #new_img = square_mask(mask)
    list = square_mask(mask, 15, 60)
    #vis_image = skimage.color.label2rgb(super_test_img)
    #pyplot.imshow(new_img[0][])
    segments = build_segment_array(test_img, list)

    for i in range(0, len(segments)):
        segments[i] = resize(segments[i], bw=False)
    return segments

    # show several examples
    fig, ax = pyplot.subplots(2, 2)

    ax[0][0].imshow(segments[0])
    ax[0][1].imshow(segments[1])
    ax[1][0].imshow(segments[2])
    ax[1][1].imshow(segments[3])

    blank = 1
    blank += 1


# iterate through brightness, histogram method like OTSU
# for alignment,