import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict

# Create Parse 
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default= 'flowers', help = 'path to flower images folder')
parser.add_argument('--save_dir', type=str, default='save_directory', help='Set directory to save checkpoints')
parser.add_argument('--arch', type=str, default= 'vgg16', help = 'Choose CNN Model Architecture')
parser.add_argument('--learning_rate', type=float, default= 0.01, help = 'Set hyperparamter: learning rates')
parser.add_argument('--hidden_units', type=int, default= 512, help = 'Set hyperparamter: hidden units')
parser.add_argument('--epochs', type=int, default= 20, help = 'Set hyperparamter: number of epochs')
parser.add_argument('--gpu', action ="store_true", help = 'Use GPU for training')  #default: no gpu
    
in_arg = parser.parse_args()
image_folder = in_arg.data_dir
cnn_model = in_arg.arch
learning_rate = in_arg.learning_rate
hidden_units = in_arg.hidden_units
epochs = in_arg.epochs
save = in_arg.save_dir

# Setup data directory for train, test and validation set
train_dir = image_folder + '/train'
valid_dir = image_folder + '/valid'
test_dir = image_folder + '/test'

# Data transformation
train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(25),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Check Torch Version & GPU availability
print("Pytorch version: ", torch.__version__)
print("GPU enabled?: ", torch.cuda.is_available())

# Use GPU if available
if in_arg.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Two different CNN models available to choose
alexnet = models.alexnet(pretrained = True)
vgg16 = models.vgg16(pretrained=True)

def classifier(cnn_model):

    # apply pre-trained cnn model
    if cnn_model == 'vgg16':
        model = vgg16
    elif cnn_model == 'alexnet':
        model = alexnet
    # freeze the parameter to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # define input feature for different model
    if cnn_model == 'vgg16':
        input_feature = 25088
    elif cnn_model == 'alexnet':
        input_feature = 9216

    # redefine classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_feature, 4096)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(0.4)),
                                            ('fc2', nn.Linear(4096,hidden_units)), 
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout(0.2)),
                                            ('fc3', nn.Linear(hidden_units, 102)),  
                                            ('output', nn.LogSoftmax(dim=1))
                                                                            ]))

    model.classifier = classifier
    
    return model

model = classifier(cnn_model)
model.to(device)
# Define loss function and optimizer
criterion = nn.NLLLoss()

# Define optimizer
# Only train the classifier parameters where not changining feature parameters
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)


# Training the model
print("*****************Training Start*******************")
train_loss = 0
test_loss = 0

for e in range(epochs):
    model.train()
    running_loss = 0
    
    # Training loop
    for images, labels in trainloader:
        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Testing loop
    model.eval()
    test_loss = 0
    accuracy = 0
            
    # Turn off gradients for testing, saves memory and computations
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    train_loss = running_loss/len(trainloader)
    test_loss = test_loss/len(testloader)

    print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(train_loss),
          "Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
print("*****************Training End*******************")    
# TODO: Save the checkpoint 
print("The trained model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())


# Define save_directory to save the checkpoint
def save_checkpoint():
    if save == "save_directory":
        checkpoint = {'input_size': input_feature,
                    'output_size': 102,
                    'hidden_layers': [4096,hidden_units],
                    'state_dict': model.state_dict(),
                    'state_optim': optimizer.state_dict(),
                    'class_to_idx': train_data.class_to_idx }
        torch.save(checkpoint, 'checkpoint.pth')
        print("The model is saved as checkpoint")
    else:
        print("The model is not saved")
    
    return checkpoint

# define input feature for different model
if cnn_model == 'vgg16':
    input_feature = 25088
elif cnn_model == 'alexnet':
    input_feature = 9216

# save the checkpoint
save_checkpoint()