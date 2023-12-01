# add all imports here, if adding new libraries BE SURE to also add to requirements.txt
import matplotlib.pyplot as pyplot

import eigenvalue_classifier.train as train
import model_classifier.model as mc

import joint_segmentation.segment as js


# test segmentation
im_array = js.demo()
classes = mc.apply_model(im_array)


#js.test()


