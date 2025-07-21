import os
from PIL import Image
import torch
import torchvision.transforms as transforms

ANOMALY_CLASSES = ['Normal', 'Pneumonia']

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_anomaly(image_tensor):
    # Placeholder: simple logic based on image mean pixel value
    # In real use, replace with a trained model
    mean_val = image_tensor.mean().item()
    # Arbitrary threshold for demonstration
    if mean_val < 0.5:
        return 'Pneumonia'
    else:
        return 'Normal'

def main():
    images_dir = 'images'
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)
            image_tensor = load_image(image_path)
            anomaly = predict_anomaly(image_tensor)
            print(f"{filename}: {anomaly}")

if __name__ == "__main__":
    main()
