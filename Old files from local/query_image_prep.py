import boto3
import aws_access
import pandas as pd 
from PIL import Image
import numpy as np 
from io import BytesIO
import cv2 
from tensorflow.keras.models import load_model
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


def read_image_from_s3_with_face(bucket, key, region_name='us-east-1'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    # print(key)
    s3_client = boto3.client('s3', aws_access_key_id = aws_access.aws_access_key_id, 
                                aws_secret_access_key = aws_access.aws_secret_access_key)

    obj = s3_client.get_object(Bucket=bucket, Key = key)
    file_stream = obj['Body']
    im = Image.open(file_stream) #.convert('L')

    # Convert into grayscale
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = gray[y:y + h, x:x + w]
        faces = cv2.resize(faces, (256,256))
    
    return np.array(faces) 


def view_image_from_s3(bucket, key, region_name='us-east-1'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3_client = boto3.client('s3', aws_access_key_id = aws_access.aws_access_key_id, 
                                aws_secret_access_key = aws_access.aws_secret_access_key)

    obj = s3_client.get_object(Bucket=bucket, Key = key)
    file_stream = obj['Body']
    im = Image.open(file_stream)
    # plt.imshow(im)
    return np.array(im)

def prepare_query_image(img):

    ### 1. Getting the list of images in the image repository
    s3_client = boto3.client('s3', aws_access_key_id = aws_access.aws_access_key_id, 
                                    aws_secret_access_key = aws_access.aws_secret_access_key)


    # List objects in the bucket (s3://hariharan-project-data/Banking-and-Finance/Face-Recognition-KYC-Tool/Data/)
    bucket = "hariharan-project-data"
    prefix = "Banking-and-Finance/Face-Recognition-KYC-Tool/Data"

    fileList = []
    for obj in s3_client.list_objects(Bucket=bucket,Prefix=prefix)['Contents']:
        files = obj['Key']
        if ('.jpg' in files) or ('.jpeg' in files) or ('.png' in files):
            fileList.append("s3://hariharan-project-data/"+files)

    # print("Number of Images: ",len(fileList))

    ### 2. Creating the fileList as a pandas dataframe and creating a target/label column
    data = pd.DataFrame(fileList, columns=['s3_img_path'])
    data['user_id'] = data['s3_img_path'].apply(lambda x: x.split('/')[6]).astype(int)
    image_data = data.copy()
    del data

    ### 3. Getting image in the form of arrays from the filePaths
    image_array_list = []

    for i in range(len(image_data)):
        s3_img_path = image_data.loc[i,'s3_img_path']
        bucket = 'hariharan-project-data'
        key = s3_img_path.split('data/')[1]
        img_array = read_image_from_s3_with_face(bucket=bucket,key=key)
        image_array_list.append(img_array)


    ### 4. Creating a Scoring set (Reference set of images for similarity matching against the query image)
    X = np.array(image_array_list)
    Y = image_data.user_id.values 

    # Creating test data (one images from all the classes)
    labels = image_data.user_id
    numClasses = len(np.unique(Y))
    idx_test = [np.where(labels == i)[0][0] for i in range(1, numClasses+1)]  
    # print("Size of scoring data: ", len(idx_test))
    y_test = Y[idx_test]
    x_test = X[idx_test]
    # print("Scoring Data: ", (x_test.shape, y_test.shape))


    ### 5. Loading the trained model
    model = load_model('output/siamese_model_75_percent_acc')

    ### 6. Treating the query image 
    imageA = img.copy()
    imageA = cv2.cvtColor(np.array(imageA), cv2.COLOR_RGB2GRAY)
    # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageA = cv2.resize(imageA,(256,256))

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(imageA, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(imageA, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = imageA[y:y + h, x:x + w]
        faces = cv2.resize(faces, (256,256))

    imageA = np.array(faces)

    ### 7. Comparing against each of the reference set and get the model predictions for similarity
    prediction_probs = []

    for i, imageB in enumerate(x_test):
        
        if i != 0:
            imageA = origA.copy()
            
        # create a copy of both the images for visualization purpose
        origA = imageA.copy()
        origB = imageB.copy()

        # # add channel a dimension to both the images
        # imageA = np.expand_dims(imageA, axis=-1)
        # imageB = np.expand_dims(imageB, axis=-1)
        
        # add a batch dimension to both images
        imageA = np.expand_dims(imageA, axis=0)
        imageB = np.expand_dims(imageB, axis=0)
        print(imageA.shape)
        print(imageB.shape)

        # scale the pixel values to the range of [0, 1]
        imageA = imageA / 255.0
        imageB = imageB / 255.0

        # use our siamese model to make predictions on the image pair,
        # indicating whether or not the images belong to the same class
        preds = model.predict([imageA, imageB])
        proba = preds[0][0]
        prediction_probs.append(proba)

    prediction_probs = [i if i >= 0.55 else 0 for i in prediction_probs]
    user_index = np.argmax(prediction_probs)
    user_id = y_test[np.argmax(prediction_probs)]

    # Checking the image of the users based on user_id from the azure sql database table 
    sel = user_id 
    s3_img_path = image_data[image_data['user_id']==sel]['s3_img_path'].values.tolist()[0]
    bucket = 'hariharan-project-data'
    key = s3_img_path.split('hariharan-project-data/')[1]
    print(key)
    img_array = view_image_from_s3(bucket=bucket,key=key)

    return user_id, img_array 



