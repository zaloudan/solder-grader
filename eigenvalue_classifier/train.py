# simple test for reading/viewing images
import matplotlib.pyplot as pyplot


def test():
    """ Test function"""
    test_img = pyplot.imread('training_data/excessive_solder/excessive (2).png')
    pyplot.imshow(test_img)
    pyplot.show()