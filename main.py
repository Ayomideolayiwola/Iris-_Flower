import pandas as pd
import numpy as np
import pickle
import streamlit as st
import sklearn


# configure the environment
st.set_page_config(page_title='Iris Flower Classification', page_icon='🙃',
                   layout='centered')

with open('rf_model.pkl', 'rb' ) as file:
    model = pickle.load(file)


def flower_model(name):

    st.title('Iris Flower Classification MOdel')
    st.write('My Model GUI')

    #input features:sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)

    sepal_length = st.text_input('sepal length (cm)', 'Enter a length between 4.3 and 7.9')
    sepal_width = st.text_input('sepal width (cm)', 'Enter a length between 2.0 and 4.4')
    petal_length = st.text_input('petal length (cm)', 'Enter a length between 1.0 and 6.9')
    petal_width = st.text_input('petal width (cm)', 'Enter a length between 0.1 and 2.5')


    # create a dataframe for the input values
    input_data = pd.DataFrame({
               'sepal length (cm)':[sepal_length],
               'sepal width (cm)':[sepal_width],
               'petal length (cm)':[petal_length],
               'petal width (cm)':[petal_width]
           })


   # predicted value
    if st.button('Predict'):
       prediction= model.predict(input_data)
       if prediction is not None and isinstance(prediction, np.ndarray): #converting it to a scalar
           prediction = prediction[0]


       #class [0:'setosa', 1:'versicolor', 2:'virginica']


       class_name = {0:'setosa', 1:'versicolor', 2:'virginica'}

       st.write(f'Prediction:{class_name[prediction]}')




if __name__ == '__main__':
    flower_model('PyCharm')