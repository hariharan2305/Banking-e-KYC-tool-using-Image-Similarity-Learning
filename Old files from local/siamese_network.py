# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D


def build_siamese_model(inputShape, embeddingDim=64):
    # specify the inputs for the feature extractor network
    inputs = Input(inputShape)

    # define the first set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.1)(x)

    # second set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.1)(x)

    # third set of CONV => RELU => POOL => DROPOUT layers
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(128)(pooledOutput)
    outputs = Dense(embeddingDim)(outputs)

    # build the model
    model = Model(inputs, outputs)

    # return the model to the calling function
    return model
