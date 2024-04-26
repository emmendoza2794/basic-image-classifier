import streamlit as st

from src.predictor import Predictor
from src.train import Train
from src.utils import Utils

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'classes_list' not in st.session_state:
    st.session_state.classes_list = None

if 'train_result' not in st.session_state:
    st.session_state.train_result = None


def predict_image():
    if st.session_state.image_file is None:
        st.error("No image file")
        return

    st.session_state.prediction_result = Predictor().predict(
        name_model=st.session_state.model,
        image=st.session_state.image_file
    )


def get_classes():
    if st.session_state.classes_list is not None:
        classes = Predictor().load_classes(
            name_model=st.session_state.model
        )

        st.session_state.classes_list = ', '.join(classes)

    else:
        st.session_state.classes_list = "There are no models to test, download the test models"

def train_model():

    st.session_state.train_result = Train().train_model(
        epochs=st.session_state.epochs,
        model_name=st.session_state.model_name
    )


st.set_page_config(
    page_title="Basic Image Classifier",
    page_icon="👋",
    layout="wide",
)

st.header('Test model', divider='rainbow')

col1, col2, col3 = st.columns([0.3, 0.5, 0.2])

with col1:
    with st.container(border=True):
        st.subheader("Image Classifier")

        st.selectbox(
            label='Select model',
            key="model",
            options=Predictor().load_list_models()
        )

        get_classes()

        st.markdown(f'**Classes:** {st.session_state.classes_list}')

        st.file_uploader(
            label="image file",
            key="image_file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
        )

        st.button(
            label="Predict category",
            type="primary",
            on_click=predict_image,
            use_container_width=True,
        )

    with st.container(border=True):
        st.write("download example models for testing")
        st.button(
            label="Download models",
            type="primary",
            on_click=Utils().download_models,
            use_container_width=True,
        )

with col2:
    with st.container(border=True, height=600):
        st.subheader("Image preview")

        if st.session_state.image_file is not None:
            st.image(
                image=st.session_state.image_file
            )

with col3:
    with st.container(border=True, height=600):
        st.subheader("Prediction result")

        if st.session_state.prediction_result is not None:

            data_result = st.session_state.prediction_result
            first_result = next(iter(data_result.items()))

            st.subheader(f":green[{first_result[0]} {round(first_result[1], 2)}%]")
            st.write(data_result)


st.header('Train model', divider='rainbow')


col4, col5 = st.columns([0.3, 0.7])

with col4:
    with st.container(border=True):
        st.subheader("Configuration")

        st.markdown('''
        :red[**Important:** to train the model with your own data or another dataset, first copy the images separated by 
        folders into the root folder "image_files". For example:] 
        ''')

        st.image("assets/image_files.png")

        st.text_input(
            label="Model name",
            value="model_test",
            key="model_name"
        )

        st.slider(
            label="Epochs",
            min_value=10,
            max_value=100,
            value=40,
            key="epochs",
            step=1
        )

        st.markdown("*At least 40 epochs are recommended to have an accuracy > 80%*")

        st.button(
            label="Train model",
            type="primary",
            on_click=train_model,
            use_container_width=True,
        )


with col5:
    with st.container(border=True, height=600):
        st.subheader("Training result")

        if st.session_state.train_result is not None:
            st.write(st.session_state.train_result)

