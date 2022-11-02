# Importing the necessary libraries
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


# Image pair generator function
def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []

    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    img_idx = [np.where(labels == i)[0] for i in range(1, numClasses+1)]
    # print(len(img_idx))

    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # print(len(label))
        # print(label)

        # Positive Pairs
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(img_idx[label-1])
        # print(idxB)
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])

        # Negative Pairs
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

