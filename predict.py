import argparse
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Create Parse 
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default= 'flowers/valid/11/image_03100.jpg', help = 'path to flower images')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Saved checkpoint in first part')
parser.add_argument('--arch', type=str, default= 'vgg16', help = 'Choose CNN Model Architecture')
parser.add_argument('--top_k', type=int, default= 3, help = 'Return top k most likely classes')
parser.add_argument('--category_names', default= "cat_to_name.json", help = 'path to json file with flower names')
parser.add_argument('--gpu', action ="store_true", help = 'Use GPU for inference')  #default: no gpu
in_arg = parser.parse_args()

# Define image path 
image_path = in_arg.input
topk = in_arg.top_k
checkpoint = in_arg.checkpoint
cnn_model = in_arg.arch

# Label Mapping
import json
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Check Torch Version & GPU availability
print("Pytorch version: ", torch.__version__)
print("GPU enabled?: ", torch.cuda.is_available())

# Use GPU if available 
if in_arg.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Two different CNN models available to choose
alexnet = models.alexnet(pretrained = False)
vgg16 = models.vgg16(pretrained=False)
    
    
# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # apply pre-trained cnn model
    if cnn_model == 'vgg16':
        model = vgg16
    elif cnn_model == 'alexnet':
        model = alexnet
    
    # Rebuild the custom classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0])),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(0.4)),
                                            ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(0.2)),
                                            ('fc3', nn.Linear(checkpoint['hidden_layers'][1], checkpoint['output_size'])),  
                                            ('output', nn.LogSoftmax(dim=1))]))
    

    model.classifier = classifier 
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']  
    
    return model

model = load_checkpoint(checkpoint)
print(model)

# Image Pre-Processing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_transforms = transforms.Compose ([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    return img_transforms(img)

# Predic the top 5 classes
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
      
    # indicate the model on evaluation mode
    model.eval()
    
    # process the image using function
    image = process_image(image_path)
    image = image.unsqueeze_(0)

    # Move the image & model to the device
    image = image.to(device)
    model = model.to(device)
  
    
    with torch.no_grad():
        log_ps = model.forward(image)
    
    # Convert log probabilities to probabilities
    ps = torch.exp(log_ps)
    
    # Obtain the top 5 predictions
    topk_probs, topk_labels = ps.topk(topk, dim=1)
    
    # Convert the topk tensors to lists
    probs = topk_probs.squeeze().tolist()
    classes = topk_labels.squeeze().tolist()
    
    return probs,classes

# Evaluate the result
probs, classes = predict(image_path, model, topk)

# Convert class numbers to class name
def get_class_name(class_num):
    return cat_to_name[str(class_num)]

class_name = [get_class_name(i) for i in classes]

# Print result 
print("The top {} probabilities are: {}".format(topk, probs))
print("The top {} predicted classes are: {} ".format(topk, class_name))

# Get the actual class & predicted class (highest prob)

actual_class_idx = os.path.basename(os.path.dirname(image_path))
actual_class = get_class_name(actual_class_idx)

predicted_class = class_name[0]

# Print the prediction vs actual class
print("\n************PREDICTION*****************")
print("The actual class: ", actual_class)
print("The predicted class (highest probability): ", predicted_class)

if actual_class == predicted_class:
    print("Congratulation! The prediction is correct! :)")
else:
    print("Opps, wrong prediction :(")