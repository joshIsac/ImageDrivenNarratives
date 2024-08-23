import os

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("TF_ENABLE_ONEDNN_OPTS:", os.environ['TF_ENABLE_ONEDNN_OPTS'])


import streamlit as st
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# Load environment variables from .env file
load_dotenv()

# Streamlit App Title and Description
st.title("Image Driven Narratives using Streamlit")
st.write("This is a simple image to text to speech generator using Streamlit, LangChain, and gTTS.")

# LangChain integration for story generation
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You generate imaginary stories based on the images."),
        ("user", "Image description: {image_description}")
    ]
)

# Initialize Groq API with API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groqApi = ChatGroq(model="llama3-70b-8192", temperature=0.2)
outputparser = StrOutputParser()
chainSec = prompt | groqApi | outputparser

# Load the image captioning model from Hugging Face
image_captioning_model_id = "salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(image_captioning_model_id)
model = BlipForConditionalGeneration.from_pretrained(image_captioning_model_id)

# Function to generate image description using BLIP model
def generate_image_description(img):
    try:
        inputs = processor(images=img, return_tensors="pt")
        outputs = model.generate(**inputs)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating image description: {str(e)}")
        return ""

# Image upload and processing
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Generate image description
        image_description = generate_image_description(img)
        st.write(f'Image Description: {image_description}')

        # Generate story based on the image description
        response = chainSec.invoke({'image_description': image_description})
        st.write('### Generated Story:')
        st.write(response)

        # Convert the generated story to audio using gTTS
        tts = gTTS(text=response, lang='en')
        audio_file = 'story_gtts.mp3'
        tts.save(audio_file)
        st.audio(audio_file)

        # Remove the audio file after playback
        os.remove(audio_file)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.write("---")
st.write("Developed using [Streamlit], [LangChain], [pyttsx3], and [gTTS].")
