# add all imports here, if adding new libraries BE SURE to also add to requirements.txt
import matplotlib.pyplot as pyplot

import eigenvalue_classifier.train as train
import cv2
import model_classifier.model as mc

import joint_segmentation.segment as js
import visualize_results.visualize as vis


# test segmentation
#vis.test()
test_img = cv2.imread('test_data/image (1).png')
locations, im_array = js.demo(test_img)

classes = mc.apply_model(im_array)
for i in range(0, len(locations)):
    locations[i][2] = classes[i]

filtered_locations = vis.list_filer(locations)

vis.apply_visualization(test_img, filtered_locations)


#js.test()


