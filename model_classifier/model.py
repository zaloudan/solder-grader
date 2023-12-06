"""
Module for running model

Please see Jupyter Notebooks on dev_cnn for training which was done through Google Colab
"""
import os.path

import numpy as np
import urllib.request as req

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image

batch_size = 64
labels = np.array(['excessive', 'insufficient', 'normal',
       'shifted', 'short'])

valtest_gen = image.ImageDataGenerator(rescale = 1./255,
                                       fill_mode='nearest')


"""
Load in learned model, trained using Colab GPUs 
SEE Jupyter Notebook on dev_cnn branch

Note: this must download from GDrive as it's >100MB
Github imposes a file size limit
"""
if not(os.path.isfile('model_classifier/resnetbase2.h5')):
    url = os.environ.get("MODEL_DOWNLOAD_URL")
    req.urlretrieve(url, "model_classifier/resnetbase2.h5")
model = load_model('model_classifier/resnetbase2.h5')


"""
Call THIS function to apply the model and

Returns MOST probable classification
"""
def apply_model(images):
    """ Call to apply model to array of image segments"""
    pred_matrix = get_predictions(images)
    return get_predictions_one(pred_matrix)


def get_predictions(images):
    """ Run images through trained model to get result,
    Note, other code, via Colab was used for training to access additional GPU power"""
    images = np.array(images)
    if len(images) == 0:
        return []

    test_data = valtest_gen.flow(images, batch_size=batch_size)
    predictions = np.array(model.predict(test_data))
    return predictions


def get_predictions_one(pred_matrix):
    """Extract most likely class for first round simple test"""

    most_likely_class = []
    for i in range(0, len(pred_matrix)):
        likely_index = np.argmax(pred_matrix[i])
        most_likely_class.append(labels[likely_index])

    return most_likely_class