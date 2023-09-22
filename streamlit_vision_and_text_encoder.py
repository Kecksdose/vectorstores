from PIL import Image
import string
import random

import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
import plotly.express as px

MODEL_NAME = "openai/clip-vit-base-patch32"


@st.cache_resource
def get_model(model_name: str = MODEL_NAME) -> CLIPModel:
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


@st.cache_resource
def get_processor(model_name: str = MODEL_NAME) -> CLIPModel:
    return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.header("Image and Text Similarity")
    st.divider()

    images = []
    images.append(Image.open("data/images/dog_lying_on_bed.jpeg"))
    images.append(Image.open("data/images/tiny_carrot_in_hand.jpeg"))

    st.subheader("Images :frame_with_picture:")

    with st.expander("Optional: Upload own image", expanded=False):
        uploads = st.file_uploader(
            "Upload own images", type=["png", "jpg"], accept_multiple_files=True
        )
        for upload in uploads:
            images.append(Image.open(upload))

    cols = st.columns([1 for _ in range(len(images))])
    for i, (col, image) in enumerate(zip(cols, images), start=1):
        with col:
            st.markdown(f"### {i}.")
            st.image(image, width=300)

    texts = []
    texts.append("tiny carrot in hand")
    texts.append("roboter from another planet")
    texts.append("dog lying on bed")

    st.subheader("Texts :memo:")
    with st.expander("Optional: Upload own texts", expanded=False):
        df_texts = pd.DataFrame({"Texts": [
            "tiny carrot in hand",
            "roboter from another planet",
            "dog lying on bed"
        ]})
        df_texts = st.data_editor(df_texts, num_rows="dynamic")
        df_texts = df_texts.dropna(subset=["Texts"])

    texts = df_texts["Texts"].to_list()
    for character, text in zip(string.ascii_uppercase, texts):
        st.markdown(f"**{character + '.'}** {text}")

    inputs = get_processor()(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    outputs = get_model()(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    st.subheader("Similarity Matrix :1234:")
    df = pd.DataFrame(probs.detach().numpy())

    fig = px.imshow(
        df,
        y=[f"<b>{i+1}.</b>" for i in range(len(images))],
        x=[f"<b>{c}.</b>" for c in list(string.ascii_uppercase[:len(texts)])],
    )
    fig.update_layout(xaxis_title="Text", yaxis_title="Image")

    st.plotly_chart(fig)
