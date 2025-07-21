import os
from flask import Flask, request, render_template_string
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

ANOMALY_CLASSES = ['Normal', 'Pneumonia']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = '''
<!doctype html>
<title>X-ray Pneumonia Detection</title>
<h2>Upload an X-ray image</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h3>Result: {{ result }}</h3>
{% endif %}
'''

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_anomaly(image_tensor):
    mean_val = image_tensor.mean().item()
    if mean_val < 0.5:
        return 'Pneumonia'
    else:
        return 'Normal'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_tensor = load_image(filepath)
            result = predict_anomaly(image_tensor)
    return render_template_string(HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
