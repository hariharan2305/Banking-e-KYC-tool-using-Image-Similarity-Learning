# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def build_siamese_model(inputShape, embeddingDim=64):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)

    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.1)(x)

    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.1)(x)

    # third set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(128)(pooledOutput)
    # outputs = BatchNormalization()(outputs)
    outputs = Dense(embeddingDim)(outputs)

    # build the model
    model = Model(inputs, outputs)

    # return the model to the calling function
    return model

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
      keepdims=True)
          
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)