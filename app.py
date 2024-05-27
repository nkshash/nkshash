import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('wound_classifier_model_googlenet.h5')
input_shape = (224, 224, 3)

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    return image_array

def predict(image):
    input_data = preprocess_image(image, (input_shape[0], input_shape[1]))
    input_data = np.expand_dims(input_data, axis=0)

    with open('./classes.txt', 'r') as file:
        class_labels = file.read().splitlines()

    predictions = model.predict(input_data)
    results = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    
    return results

# Create a Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.inputs.Image(type="pil"), 
    outputs=gr.outputs.Label(num_top_classes=18),
    live=True
)

# Launch the Gradio interface
iface.launch()
