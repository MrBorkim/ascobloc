from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('final xgboost')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == 'Online':

        Długość = st.number_input('Długość', min_value=100, max_value=3000, value=1200)
        Głębokość = st.number_input('Głębokość', min_value=100, max_value=3000, value=600)
        Wysokość = st.number_input('Wysokość', min_value=50, max_value=2500, value=850)
        
        if st.checkbox('Standard'):
            smoker = 'yes'
        else:
            smoker = 'no'
        Typ = st.selectbox('Typ katalogowy', ['A3100.126', 'A3100.128', 'A3101.126', 'A3105.126'])

        output=""

        input_dict = {'Długość' : Długość, 'sex' : Głębokość, 'Głębokość' : bmi, 'Wysokość' : Wysokość,  'Typ katalogowy' : Typ}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
