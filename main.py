# add all imports here, if adding new libraries BE SURE to also add to requirements.txt
import matplotlib.pyplot as pyplot

import eigenvalue_classifier.train as train


# simple test for reading/viewing images
test_img = pyplot.imread('training_data/excessive_solder/excessive (1).png')
pyplot.imshow(test_img)
pyplot.show()

train.test()



