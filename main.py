import os
from PIL import Image
import torch
import torchvision.transforms as transforms

# Placeholder anomaly names
ANOMALY_CLASSES = ['Normal', 'Pneumonia', 'Tuberculosis', 'Other']

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_anomaly(image_tensor):
    # Placeholder: randomly select an anomaly
    idx = torch.randint(0, len(ANOMALY_CLASSES), (1,)).item()
    return ANOMALY_CLASSES[idx]

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
