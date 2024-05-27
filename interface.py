import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transformers import pipeline
import requests
import io
font_path = "Arial.ttf"

font = ImageFont.truetype(font_path, 16)

def load_model(repo_id):
    """
    Loads the image classification pipeline from the specified repository.

    This function is cached to improve performance by loading the model only once.

    Args:
        repo_id (str): ID of the Hugging Face Hub repository containing the model.

    Returns:
        transformers.pipelines.ImageClassificationPipeline: The loaded pipeline.
    """
    return pipeline("image-classification", model=repo_id)

import requests

API_URL_MALIGNANT = "https://api-inference.huggingface.co/models/karoladelk/swin-tiny-patch4-window7-224-finetuned-eurosat"
API_URL_CLASSIFICATION = "https://api-inference.huggingface.co/models/karoladelk/swin-tiny-patch4-window7-224-classification"
headers = {"Authorization": "Bearer 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"}

import io

def classify_image(image):
  """
  Classifies an image using the provided API.

  Args:
      image (PIL.Image): Image object for classification.

  Returns:
      list: List of dictionaries containing class labels and scores, or None if error occurs.
  """
  try:
    # Convert image to bytes format (assuming RGB mode)
    image_bytes = io.BytesIO()
    image.convert('RGB').save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response_skin_cancer = requests.post(API_URL_MALIGNANT, headers=headers, data=image_bytes)

    predicted_mal=response_skin_cancer.json()[0]['label']
    if predicted_mal=="Malignant" or predicted_mal=="Benign":
        image_bytes.seek(0)
        response_classification = requests.post(API_URL_CLASSIFICATION, headers=headers, data=image_bytes)
        if response_classification.json()[0]['score']>0.0:
            predicted_mal = response_classification.json()[0]['label']
        else:
            predicted_mal = "Healthy"
        # st.write(response_classification.json()[0]['label'])
        # st.write(response_classification.json()[0])
        # predicted_mal=response_classification.json()[0]['label']

    return predicted_mal



  except Exception as e:
    st.error(f"Error in classification: {e}")
    return None

def preprocess_image(image):
    """
    Preprocesses an image for classification.

    Args:
        image (PIL.Image): Image to preprocess.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Convert the image to RGB if it's not already in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the required input size (e.g., 224x224)
    image = image.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.tensor(image_array)

    # Add batch dimension to the tensor
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def object_detection_yolo(image):
    """
    Performs object detection using YOLO model.

    Args:
        image (PIL.Image): Image for object detection.

    Returns:
        list: List of tuples containing cropped images, bounding box coordinates, and class labels.
    """
    # API endpoint and headers for YOLO model
    YOLO_URL = "https://api.ultralytics.com/v1/predict/LP9vwrVH0sakQnw3DV7C"
    headers_YOLO = {"x-api-key": "e1e10ed2383a82c8b88fb040ed2951126e9a7d850e"}

    data_YOLO = {"size": 640, "confidence": 0.15, "iou": 0.999}

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    try:
        # Upload image to YOLO model for object detection
        response_YOLO = requests.post(YOLO_URL, headers=headers_YOLO, data=data_YOLO, files={"image": image_bytes})
        response_YOLO.raise_for_status()  # Raise an exception if YOLO request fails
        yolo_results = response_YOLO.json()['data']
    except Exception as e:
        st.error(f"Error in object detection: {e}")
        return []

    # Process YOLO results
    detections = []
    for bbox in yolo_results:
        xcenter = int(bbox['xcenter'] * image.width)
        ycenter = int(bbox['ycenter'] * image.height)
        width = int(bbox['width'] * image.width)
        height = int(bbox['height'] * image.height)
        xmin = xcenter - width // 2
        ymin = ycenter - height // 2
        xmax = xcenter + width // 2
        ymax = ycenter + height // 2

        # Check if the bounding box contains a 'label' key
        if 'label' in bbox:
            class_label = bbox['label']
        else:
            class_label = "Unknown"

        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        detections.append((cropped_image, (xmin, ymin, xmax, ymax), class_label))

    return detections

def calculate_text_size(text, font):
    # Draw temporary text to get the bounding box
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

st.title("Skin Cancer Image Classification")
uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image", use_column_width=True)

    detections = object_detection_yolo(image)

    if detections:
        image_with_draw = image.copy()
        draw = ImageDraw.Draw(image_with_draw)

        st.write("Classification Results:")
        for i, (cropped_image, bbox, class_label) in enumerate(detections):
            # Classify the cropped image
            prediction = classify_image(cropped_image)
            text_width, text_height = calculate_text_size(prediction, font)

            text_position = (bbox[0] + (bbox[2] - bbox[0]) // 2 - text_width // 2, bbox[1] - text_height - 5)
            # Draw bounding box
            if prediction == "Healthy":
                draw.rectangle(bbox, outline="green", width=2)
                draw.text(text_position, prediction, fill="green", font=font)
            else:
                draw.rectangle(bbox, outline="red", width=2)
                draw.text(text_position, prediction, fill="red", font=font)
        st.image(image_with_draw,  use_column_width=True)