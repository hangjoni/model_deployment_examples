import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import read_image



imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]



def predict():
    file = 'kitten.jpg'#'Original-image-256x256-pixels.png'
    input_tensor = transform_image(file)
    class_id, class_name = get_prediction(input_tensor)
    return class_name


res = predict()
print(res)
