import streamlit as st
from PIL import Image
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img,img_to_array


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;margin-top: -40px;margin-bottom: 60px;'>Cardiovascular Disease Predictor</h1>", unsafe_allow_html=True)
container = st.container
col1, col2, col3, col4 = st.columns(4)
with col1:
    uploaded_file1 = st.file_uploader("Choose an iamge of systole high pressure", type=["jpg","jpeg"])
    if uploaded_file1 is not None:
        image1 = Image.open(uploaded_file1)
        image1.save('image1.jpg')
        st.image(image1, caption='Uploaded Image.',use_column_width=True)

with col2:
    uploaded_file2 = st.file_uploader("Choose an image of systole low pressure", type=["jpg","jpeg"])
    if uploaded_file2 is not None:  
        image2 = Image.open(uploaded_file2)
        image2.save('image2.jpg')
        st.image(image2, caption='Uploaded Image.', use_column_width=True)

with col3:
    uploaded_file3 = st.file_uploader("Choose an image of diastole high pressure", type=["jpg","jpeg"])
    if uploaded_file3 is not None:
        image3 = Image.open(uploaded_file3) 
        image3.save('image3.jpg')
        st.image(image3, caption='Uploaded Image.', use_column_width=True)

with col4:
    uploaded_file4 = st.file_uploader("Choose an image of diastole low pressure", type=["jpg","jpeg"])
    if uploaded_file4 is not None:
        image4 = Image.open(uploaded_file4)
        image4.save('image4.jpg')
        st.image(image4, caption='Uploaded Image.', use_column_width=True)
st.markdown('#')

col1, col2, col3, col4, col5 = st.columns(5)


with col1:
    age = st.text_input('Enter the age:')

with col2:
    gender = st.selectbox('Pick your gender:', ["Male",'Female'])
    gen = 0
    if(gender == 'Female'):
        gen = 1
        
with col3:
    ap_hi = st.text_input('Enter the systolic blood pressure:')
    # try:
    #     ap_hi1 = int(ap_hi)
    # except ValueError:
    #     st.write("Please enter a valid integer.")

with col4:
    ap_lo = st.text_input('Enter the diastolic blood pressure:')
    # try:
    #     ap_lo1 = int(ap_lo)
    # except ValueError:
    #     st.write("Please enter a valid integer.")
with col5:
    chol = st.selectbox('Enter the cholestrol level:',["Normal",'Above normal','Well above normal'])
    chol1 = 1
    if chol == 'Above normal':
        chol1 = 2
    if chol == 'Well above normal':
        chol1 = 3

    # try:
    #     chol1 = int(chol)
    # except ValueError:
    #     st.write("Please enter a valid integer.")

col6, col7, col8, col9, col10    = st.columns(5)
with col6:
    gluc = st.selectbox('Enter the glucose level:',["Normal",'Above Normal','Well above normal'])
    gluc1 = 1
    if gluc == 'Above normal':
        gluc1 = 2
    if gluc == 'Well above normal':
        gluc1 = 3
    # try:
    #     gluc1 = int(gluc)
    # except ValueError:
    #     st.write("Please enter a valid integer.")

with col7:
    smoke_st = st.selectbox('Smoking status:', ["Yes",'No'])
    sm = 0
    if(sm == 'No'):
        sm = 1

with col8:
    alc_st = st.selectbox('Alcohol consumption status:', ["Yes",'No'])
    alc = 0
    if(alc == 'No'):
        alc = 1
with col9:
    phy_act = st.selectbox('Physical activity status:', ["Yes",'No'])
    phy = 0
    if(phy == 'No'):
        phy = 1
with col10:
    bmi = st.text_input('Enter the bmi:')
    # try:
    #     bmi1 = float(bmi)
    # except ValueError:
    #     st.write("Please enter a valid float value.")

# lst = [age1,gen,ap_hi1,ap_lo1,chol1,gluc,alc_st,phy,bmi1]

st.markdown('#')
# col11, col12, col13    = st.columns([2,1,2])

# with col12:
#     col14,col15,col16 = st.columns(3)
#     with col15:

if st.button('Predict'):
    # st.write('write your code here')
    if uploaded_file1 is None:
        st.warning('Please upload image of systole high pressure')
    elif uploaded_file2 is None:
        st.warning('Please upload image of systole low pressure')
    elif uploaded_file3 is None:
        st.warning('Please upload image of diastole high pressure')
    elif uploaded_file4 is None:
        st.warning('Please upload image of diastole low pressure')
    else:
        import numpy as np
        test_model=load_model('mymodel2.h5')
        img1=load_img('image1.jpg', target_size=(224,224))
        img2=load_img('image2.jpg', target_size=(224,224))
        img3=load_img('image3.jpg', target_size=(224,224))
        img4=load_img('image4.jpg', target_size=(224,224))

        pred = []
         
        x1=img_to_array(img1)
        x1=np.expand_dims(x1, axis=0)
        img_dat1=preprocess_input(x1)
        classes1=test_model.predict(img_dat1)
        temp1 = list(classes1)
        # st.write(temp1)

        # st.write(int(temp1[0][1]))
        if temp1[0][0] == 1:
            pred.append(0)
        else:
            pred.append(1)


        x2=img_to_array(img2)
        x2=np.expand_dims(x2, axis=0)
        img_dat2=preprocess_input(x2)
        classes2=test_model.predict(img_dat2)
        temp2 = list(classes2)
        # st.write(temp2)
        # st.write(int(temp1[0][1]))
        if temp2[0][0] == 1:
            pred.append(0)
        else:
            pred.append(1)

        x3=img_to_array(img3)
        x3=np.expand_dims(x3, axis=0)
        img_dat3=preprocess_input(x3)
        classes3=test_model.predict(img_dat3)
        temp3 = list(classes3)
        # st.write(temp3)


        # st.write(int(temp1[0][1]))
        if temp3[0][0] == 1:
            pred.append(0)
        else:
            pred.append(1)

        x4=img_to_array(img4)
        x4=np.expand_dims(x4, axis=0)
        img_dat4=preprocess_input(x4)
        classes4=test_model.predict(img_dat4)     
        temp4 = list(classes4)
        # st.write(temp4)


        # st.write(int(temp1[0][1]))
        if temp4[0][0] == 1:
            pred.append(0)
        else:
            pred.append(1)  
        
        # st.write(pred)
        try:
            age1 = int(age)
        except ValueError:
            # st.write("Please enter a valid integer.")
            st.warning('Please enter a valid integer for age.')
        try:
            ap_hi1 = int(ap_hi)
        except ValueError:
            st.warning("Please enter a valid integer systolic blood pressure.")
        try:
            ap_lo1 = int(ap_lo)
        except ValueError:
            st.warning("Please enter a valid integer diastolic blood pressure.")              
        try:
            chol2 = int(chol1)
        except ValueError:
            st.warning("Please enter a valid integer colestrol.")   
        try:
            gluc2 = int(gluc1)
        except ValueError:
            st.warning("Please enter a valid integer glucose.")
        try:            
            bmi1 = float(bmi)
        except ValueError:
            st.warning("Please enter a valid float value of bmi.")
            
        values = []
        values.append(age1)
        values.append(gen)
        values.append(ap_hi1)
        values.append(ap_lo1)
        values.append(chol2)
        values.append(gluc2)
        values.append(sm)
        values.append(alc)
        values.append(phy)
        values.append(bmi1)
        # st.write(values)
        final_val = np.array(values)
        final_val = final_val.reshape(1,-1)

        pickled_model = pickle.load(open('model_health.pkl', 'rb'))
        prediction =  pickled_model.predict(final_val)

        if(prediction == 0):
            pred.append(0)
        else:
            pred.append(1)
        ct = pred.count(1)
        if ct > 2:
            st.markdown("<h5 style='color:red;margin-top: 10px;margin-bottom: 60px;'>Cardiovascular disease is detected</h5>", unsafe_allow_html=True)
        else:
            st.markdown("<h5 style='margin-top: 10px;margin-bottom: 60px;'>Cardiovascular disease is not detected</h5>", unsafe_allow_html=True)



 




