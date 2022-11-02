from PIL import Image
import boto3
from io import BytesIO
import numpy as np
import aws_access
import matplotlib.pyplot as plt
import cv2


s3_client = boto3.client('s3', aws_access_key_id=aws_access.aws_access_key_id,
                         aws_secret_access_key=aws_access.aws_secret_access_key)


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
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    file_stream = obj['Body']
    im = Image.open(file_stream)  # .convert('L')
    print(type(im))
    # plt.imshow(im)
    # print(np.array(im).shape)

    # Convert into grayscale
    gray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = gray[y:y + h, x:x + w]
        cv2.imshow("face", faces)
        plt.imshow(faces)
        # cv2.imwrite('face.jpg', faces)

    # return np.array(faces)
