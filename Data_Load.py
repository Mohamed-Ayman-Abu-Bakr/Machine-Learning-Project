import numpy as np
from samples import *

def loadDigitsData(sampleSize, testSize):
    digits_labels_train = loadLabelsFile('digitdata/traininglabels', sampleSize)
    digits_labels_test = loadLabelsFile('digitdata/testlabels', testSize)
    datum_train = loadDataFile('digitdata/trainingimages', sampleSize, 28, 28)
    datum_test = loadDataFile('digitdata/testimages', testSize, 28, 28)

    digits_train = []
    digits_test = []
    for i in range(len(datum_train)):
        digits_train.append(datum_train[i].getPixels())
    for i in range(len(datum_test)):
        digits_test.append(datum_test[i].getPixels())

    digits_labels_train = np.array(digits_labels_train)
    digits_labels_test = np.array(digits_labels_test)
    digits_train = np.array(digits_train).reshape(sampleSize, 28*28)
    digits_test = np.array(digits_test).reshape(testSize, 28*28)

    return digits_labels_train, digits_labels_test, digits_train, digits_test



def loadFacesData(sampleSize, testSize):
    faces_labels_train = loadLabelsFile('facedata/facedatatrainlabels', sampleSize)
    faces_labels_test = loadLabelsFile('facedata/facedatatestlabels', testSize)
    datum_train = loadDataFile('facedata/facedatatrain', sampleSize, 60, 60)
    datum_test = loadDataFile('facedata/facedatatest', testSize, 60, 60)

    faces_train = []
    faces_test = []
    for i in range(len(datum_train)):
        faces_train.append(datum_train[i].getPixels())
    for i in range(len(datum_test)):
        faces_test.append(datum_test[i].getPixels())

    faces_labels_train = np.array(faces_labels_train)
    faces_labels_test = np.array(faces_labels_test)
    faces_train = np.array(faces_train).reshape(sampleSize, 60*60)
    faces_test = np.array(faces_test).reshape(testSize, 60*60)

    return faces_labels_train, faces_labels_test, faces_train, faces_test