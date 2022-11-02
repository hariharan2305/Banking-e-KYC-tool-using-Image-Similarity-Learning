from sqlite3 import register_adapter
import streamlit as st 
from PIL import Image
import cv2 
from query_image_prep import * 
import matplotlib.pyplot as plt 
import io
import aws_access

st.markdown('# Welcome to the Banking e-KYC Tool')


# Loading the logo for the tool
logo = cv2.imread('projectPrologo.jpeg')
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
logo = cv2.resize(logo,(400,400))
st.sidebar.image(logo)

def uid_check(uid):
    if uid in uid_in_repo:
        st.write('This USER_ID is already taken! Please create a new one.')
        return 'Already Present'
    else:
        return 'New User'
    

# Functions of the tool
# 1. For an existing user --> Get the query image and use the trained model to get the matching pair and retrieve the details of the user
# 2. For a new user --> Get the  details of the user along with the sample images. Add the details to the sql database and image to s3 bucket
# 3. Set the threshold to be 60% and if no matches above 60% comes for a search, then ask the user if he/she is new or old user. If old, then ask for their name and do a name search. If new, then ask them to upload the image.


# Checking if the user is a new or old
# st.write('##### Hey there... Are you an existing user?')
user_ind = st.radio(label='Hey there... Are you an existing user?',options = ['Yes','No'])

if user_ind == 'Yes':
    st.write('')
    st.write('##### Welcome!!')
    # st.write('Please upload your image for verification:')
    img_up = st.radio(label='Please upload your image for verification:',options=['Upload Image','Take Photo'])

    if img_up == 'Upload Image':
        file = st.file_uploader(label='')
    else:
        file = st.camera_input(label='')

    # st.image(file)
    file = Image.open(file)
    st.write('###### Query Image:')
    st.image(file)
    # st.write(type(file))

    if st.button(label='Find a match'):
        output_uid, output_img = prepare_query_image(file)
        st.write('##### The matched User ID and Image:')
        # st.write('###### Matched Image from the Repository:')
        st.write(f'USER_ID: {output_uid}')
        st.image(output_img)
    


if user_ind == 'No':
    st.write('Please enter your details and upload your images.')
    st.markdown('##### Enter your details:')

    
    # List objects in the bucket (s3://hariharan-project-data/Banking-and-Finance/Face-Recognition-KYC-Tool/Data/)
    s3_client = boto3.client('s3', aws_access_key_id = aws_access.aws_access_key_id, 
                                aws_secret_access_key = aws_access.aws_secret_access_key)

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
    uid_in_repo = data.user_id.unique().tolist()

    uid = int(st.number_input(label='Enter your USER_ID:',min_value=0,format='%d'))
    uid_button = st.button(label='Check USER_ID availablitiy')
    uid_check_ind = ''
    if uid_button:
        uid_check_ind = uid_check(uid)

# if uid_check_ind == 'New User':

    img_up = st.radio(label='Please upload your image for registration:',options=['Upload Image','Take Photo'])

    if img_up == 'Upload Image':
        file = st.file_uploader(label='')
    else:
        file = st.camera_input(label='')

    register_button = st.button(label='Register')

    if register_button:
        pil_img = Image.open(file)

        # Save the image to an in-memory file
        in_mem_file = io.BytesIO()
        pil_img.save(in_mem_file, format=pil_img.format)
        in_mem_file.seek(0)

        bucket = "hariharan-project-data"
        prefix = f"Banking-and-Finance/Face-Recognition-KYC-Tool/Data/{uid}/{uid}_1.jpeg" 

        s3_client.upload_fileobj(in_mem_file,bucket,prefix,ExtraArgs={ "ContentType": "image/jpeg"})

        st.write('User registration is successful')


    