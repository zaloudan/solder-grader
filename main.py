# add all imports here, if adding new libraries BE SURE to also add to requirements.txt
import os

import matplotlib.pyplot as pyplot

import eigenvalue_classifier.train as train
import cv2
import model_classifier.model as mc

import joint_segmentation.segment as js
import visualize_results.visualize as vis

""

img_path = os.environ.get("IN_IMG_PATH")
test_img = cv2.imread(img_path)
locations, im_array = js.segment_image(test_img)


"""
Apply the trained model 

"""
classes = mc.apply_model(im_array)
for i in range(0, len(locations)):
    locations[i][2] = classes[i]

filtered_locations = vis.list_filer(locations)

"""
Call visualization module to display the graded work

NOTE: the result will be saved as "graded_work.jpg"
"""
vis.apply_visualization(test_img, filtered_locations)

# End driver program, image will be saved