import streamlit as st
import os
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import numpy as np

# Funktion zur Durchführung von Vorhersagen mit dem YOLO-Modell
def predicition_yolo(yolo_path: str, image_path: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_yolo = YOLO(yolo_path)

    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_yolo(image_tensor)
        # Verwendung von top1 für die Vorhersage
        predicted_class = outputs[0].probs.top1

    return predicted_class

# Hauptfunktion der Streamlit-App
def main():
    st.set_page_config(page_title="Hautläsion Klassifikation", page_icon="🩺", layout="wide")
    
    st.title('🔍 Hautläsion Klassifikation')
    st.write("Willkommen zu unserer **Hautläsion Klassifikations-App**! Nutzen Sie diese App, um zu überprüfen, ob eine Hautläsion bösartig oder gutartig ist.")
    
    st.sidebar.header("Über diese App")
    st.sidebar.write("""
    Diese App verwendet ein vortrainiertes YOLO-Modell zur Klassifikation von Hautläsionen.
    Laden Sie einfach ein Bild hoch oder wählen Sie ein bereits vorhandenes Bild aus, um eine Vorhersage zu erhalten.
    """)
    
    # Verzeichnis mit den vorhandenen Bildern
    image_dir = r'F:\KI in den Life Sciences\hautkrebserkennung\YOLO\images\test'
    categories = ['boesartig', 'gutartig']

    image_files = []
    for category in categories:
        category_path = os.path.join(image_dir, category)
        if os.path.exists(category_path):
            files = [os.path.join(category, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            image_files.extend(files)

    # Layout mit zwei Spalten
    col1, col2 = st.columns(2)

    with col1:
        selected_image = st.selectbox("Wählen Sie ein Bild aus den hochgeladenen Bildern aus:", image_files)
        if selected_image:
            image_path = os.path.join(image_dir, selected_image)
            image = Image.open(image_path)
            st.image(image, caption='Ausgewähltes Bild.', use_column_width=True)

    with col2:
        uploaded_file = st.file_uploader("Oder laden Sie ein neues Bild hoch:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Hochgeladenes Bild.', use_column_width=True)
            image_path = os.path.join(image_dir, uploaded_file.name)
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
    
    if selected_image or uploaded_file:
        if st.button("Vorhersagen"):
            with st.spinner("Vorhersage wird durchgeführt..."):
                try:
                    YOLO_PATH = r'F:\KI in den Life Sciences\hautkrebserkennung\runs\classify\train3\weights\best_for_2_classes.pt'  # Pfad zum gespeicherten YOLO-Modell
                    prediction = predicition_yolo(YOLO_PATH, image_path)
                    labels = ["Gutartig", "Bösartig"]
                    st.success(f"**Vorhersage:** {labels[prediction]}")
                except Exception as e:
                    st.error(f"Fehler bei der Vorhersage: {e}")

if __name__ == '__main__':
    main()