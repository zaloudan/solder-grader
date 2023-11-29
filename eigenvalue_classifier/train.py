# simple test for reading/viewing images
import matplotlib.pyplot as pyplot
import numpy as np
import skimage.transform
import cv2
import os

def import_train_data():
    """ This code handles importing the test data"""
    categories = [  ['normal', 'training_data/normal/'],
                    ['excessive', 'training_data/excessive_solder/'],
                    ['insufficient', 'training_data/insufficient_solder/'],
                    ['shifted', 'training_data/shifted_component/'],
                    ['short', 'training_data/short/']]

    keys = []
    filenames = []
    for category in categories:
        list = os.listdir(category[1])
        for file in list:
            keys.append(category[0])
            filenames.append(category[1] + file)

    # load in all files
    train_data = []
    for file in filenames:
        train_data.append(cv2.imread(file))

    return keys, train_data

def resize(img, bw=False):
    """ Convert images to standard size"""
    rows, cols, colors = img.shape
    dim = max(rows, cols)

    new_image = np.zeros((dim, dim, 3), dtype=np.uint8)

    # determine upper-left corner
    r_off = int((dim - rows)/2)
    c_off = int((dim - cols)/2)

    new_image[r_off:r_off+rows,c_off:c_off+cols,:] = img

    if bw:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    return skimage.transform.resize(new_image, (28,28))

def preprocess_set(img_array):
    """ Preprocess set of images """
    # preprocess training images
    for i in range(0, len(img_array)):
        img_array[i] = resize(img_array[i], bw=True)
        img_array[i] = np.ndarray.flatten(img_array[i])

    return img_array

def neighbor_distances(train, test):
    """ Find distances to neighbors in eigenspace"""

    distances = np.zeros(shape=(len(train), len(test)))

    # iterate over training data
    for i in range(0, len(train)):
        for j in range(0, len(test)):
            distances[i, j] = np.linalg.norm(train[i, :] - test[j, :])

    return distances

def classify(keys, neighbor_distances, num_matches=3):
    """ Classify images using some method"""
    best_matches = np.empty(shape=(len(neighbor_distances[0, :]), num_matches), dtype=object)
    neighbor_distances = np.transpose(neighbor_distances)

    for i in range(0, len(neighbor_distances[:, 0])):
        img = neighbor_distances[i, :]
        max_distance = np.max(img)
        for j in range(0, num_matches):
            min_idx = np.argmin(img)
            best_matches[i, j] = keys[min_idx]
            img[min_idx] = max_distance

    return best_matches

def preprocess():
    train_keys, train_imgs = import_train_data()
    # train_imgs = []
    # train_imgs.append(cv2.imread('training_data/normal/normal_b2 (1).tiff'))
    # train_imgs.append(cv2.imread('training_data/normal/normal_b2 (3).tiff'))
    # train_imgs.append(cv2.imread('training_data/normal/normal_b2 (4).tiff'))
    # train_imgs.append(cv2.imread('training_data/excessive_solder/excessive (1).png'))

    train_imgs = preprocess_set(train_imgs)

    # normalize
    train_mean = np.mean(train_imgs, axis=0)

    train_array = np.array(train_imgs) - train_mean
    train_array_t = np.transpose(train_array)
    eMatrix = np.matmul(train_array_t, train_array)

    evals, evecs = np.linalg.eig(eMatrix)

    # trim eigenvectors
    final_evecs = evecs[:,0:50]
    train_ebasis = np.matmul(train_array, final_evecs)

    # test data
    test_data = []
    test_data.append(cv2.imread('training_data/normal/normal_b2 (7).tiff'))
    test_data.append(cv2.imread('training_data/normal/normal_b2 (8).tiff'))
    test_data.append(cv2.imread('training_data/excessive_solder/excessive (3).png'))

    test_data = preprocess_set(test_data)
    test_array = np.array(test_data) - train_mean
    test_ebasis = np.matmul(test_array, final_evecs)

    distance_matrix = neighbor_distances(train_ebasis, test_ebasis)
    matches = classify(train_keys, distance_matrix)

    test = 0
    test = test




def test():
    """ Test function"""
    # test_img = pyplot.imread('training_data/excessive_solder/excessive (2).png')
    # pyplot.imshow(test_img)
    # pyplot.show()
    preprocess()