import matplotlib.pyplot as pyplot
import cv2
import numpy
import numpy as np
import skimage.color
from skimage import segmentation



def test():
    """Test function to show image"""
    test_img = pyplot.imread('test_data/image (1).png')
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    pyplot.imshow(gray_img)
    pyplot.show()


def split_mask(mask):
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

def square_mask(mask, boundary):
    result = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    return_img = np.zeros(mask.shape)
    crop_list = np.zeros(shape=(100, 4), dtype=int)

    counter = 0
    for region in result:
        pixels, blank1, blank2 = region.shape
        if pixels > 25:
            crop_list[counter][0] = int(min(region[:, :, 1])) - boundary
            crop_list[counter][1] = int(max(region[:, :, 1])) + boundary
            crop_list[counter][2] = int(min(region[:, :, 0])) - boundary
            crop_list[counter][3] = int(max(region[:, :, 0])) + boundary

            # for row in range(crop_list[counter][0], crop_list[counter][1]):
            #     for col in range(crop_list[counter][2], crop_list[counter][3]):
            #         return_img[row][col] = counter
            counter += 1

    return crop_list
    return return_img


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
    test_img_hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
    light_gray = (0, 0, 145)
    dark_gray = (255, 40, 255)
    mask = cv2.inRange(test_img_hsv, light_gray, dark_gray)
    mask = cv2.medianBlur(mask, 7)
    pyplot.imshow(mask, cmap="gray")

    res_img = test_img.shape
    res_img = cv2.bitwise_or(test_img, res_img, mask=mask)
    #pyplot.imshow(res_img, cmap='hsv')

    # need to find the centers of the objects
    #new_img = square_mask(mask)
    list = square_mask(mask, 5)
    #vis_image = skimage.color.label2rgb(super_test_img)
    #pyplot.imshow(new_img[0][])
    sub = list[7]
    pyplot.imshow(test_img[sub[0]:sub[1], sub[2]:sub[3], :])

    blank = 1
    blank += 1