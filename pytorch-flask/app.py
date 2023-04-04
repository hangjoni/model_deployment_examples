import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image
from flask import Flask, request, jsonify


app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))

# Load pre-trained model from PyTorch Hub
model = models.densenet121(pretrained=True)
model.eval()


# Transform input into the form our model expects
def transform_image(image_bytes):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms) 
    img = Image.open(io.BytesIO(image_bytes))                          # Open the image file
    timg = my_transforms(img)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        # Loading image from local folder
        # with open("./kitten.jpg", 'rb') as f:
        #     image_bytes = f.read()

        # Accepting image via POST request
        file = request.files['file']
        # convert that to bytes
        image_bytes = file.read()

        input_tensor = transform_image(image_bytes)
        class_id, class_name = get_prediction(input_tensor)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
