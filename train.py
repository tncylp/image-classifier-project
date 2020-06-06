import numpy as np
import time
import os
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from workspace_utils import active_session

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Image Classifier - train.py')
    
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'choice of model architecture')
    parser.add_argument('--data_dir', type = str, default = "flowers", help = 'directory for the data set')
    parser.add_argument('--save_dir', type = str, help = 'directory for the model will be saved')
    parser.add_argument('--hidden_units', type = int, default = 4000, help = 'number of hidden units')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'value of learning rate')
    parser.add_argument('--epochs', type = int, default = 6, help = 'number of epochs')
    parser.add_argument('--gpu', action = "store_true", help = 'GPU enabled instead of CPU?')
    

    return parser.parse_args()

#----------------------------------------------------------------------------------

def load_data(data_dir = 'flowers'):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform = data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform = data_transforms['testing'])
    }


    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64)
    }
    
    return dataloaders['training'], dataloaders['validation'], dataloaders['testing'], image_datasets['training']

#----------------------------------------------------------------------------------

def set_model(arch = 'vgg16'):
    
    model = getattr(models, arch)(pretrained = True)
    model.name = arch
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

#----------------------------------------------------------------------------------

# Class for defining a new network as classifier
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        
        super().__init__()
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Add dropout probability
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x
    
#----------------------------------------------------------------------------------

def set_classifier(input_size = 25088, output_size = 102, hidden_units = 4000, drop_out = 0.5):
    
    hidden_layers = [hidden_units, 1000]
    
    return Classifier(input_size, output_size , hidden_layers, drop_out)

#----------------------------------------------------------------------------------

def get_device(gpu_enabled):
    
    return torch.device("cuda:0" if gpu_enabled else "cpu")

#----------------------------------------------------------------------------------

# Function for the validation
def validate(model, criterion, testloader, device):
    
    # Set hardware config (to GPU or CPU)
    model.to(device)
    
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        
        # Move input and label tensors to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model.forward(inputs)
        
        # Compute loss
        test_loss += criterion(outputs, labels).item()

        # Softmax distribution
        ps = torch.exp(outputs)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

#----------------------------------------------------------------------------------

# Function for the train models
def train(model, epochs, criterion, optimizer, trainloader, validloader, device):
    
    # Set hardware config (to GPU or CPU)
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 1
    
    # Looping in epochs
    for epoch in range(epochs):
        
        # Set to training mode
        model.train()

        # Looping in images
        for inputs, labels in trainloader:

            steps += 1

            # Move input and label tensors to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset the existing gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update weights
            optimizer.step()

            # Compute the total loss for the batch
            running_loss += loss.item()

        # DONE: Track the loss and accuracy on the validation set to determine the best hyperparameters
        if steps % print_every == 0:

            # Set to evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():

                # Validate model
                test_loss, accuracy = validate(model, criterion, validloader, device)
                
            training_loss = running_loss/print_every
            validation_loss = test_loss/len(validloader)
            validation_accuracy = accuracy/len(validloader)

            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(training_loss),
                  "Test Loss: {:.3f}.. ".format(validation_loss),
                  "Test Accuracy: {:.3f}".format(validation_accuracy))

            running_loss = 0
            
            # Set back to the training mode
            model.train()
                
    return training_loss, validation_loss, validation_accuracy

#----------------------------------------------------------------------------------

def save_checkpoint(model, training_dataset, save_dir, checkpoint_name = 'checkpoint1'):
    
    model.class_to_idx = training_dataset.class_to_idx
    
    checkpoint = {
        'arch': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }
    
    if not save_dir is None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = save_dir + os.path.sep + checkpoint_name + '.pth'
    else:
        file_name = checkpoint_name + '.pth'
    
    torch.save(checkpoint, file_name)
    
    return file_name

#----------------------------------------------------------------------------------

def main():
    
    args = get_input_args()
    print("Arguments have been parsed..")
    
    trainloader, validloader, testloader, training_dataset = load_data(args.data_dir)
    print("Data loaders have been loaded..")
    
    model = set_model(args.arch)
    print("Model has been set up..")
    model.classifier = set_classifier(hidden_units = args.hidden_units)
    print("Classifier has been set up..")
    
    lr = args.learning_rate
    print("Learning rate: {}".format(lr))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    print("Criterion is set to 'NLLLoss' and optimizer is set to 'Adam'")
    
    print("GPU enabled? {}".format(args.gpu))
    device = get_device(args.gpu)
    print("Device: {}".format(device))
    model.to(device)
    
    epochs = args.epochs
    print("Epochs: {}".format(epochs))
    
    print("Training has been started.. please wait, this can take a while..")
    with active_session():
        training_loss, valid_loss, valid_accuracy = train(model, epochs, criterion, optimizer, trainloader, validloader, device)
    
    print("\n***\n\nTrain Loss: {}\nTest Loss: {}\nTest accuracy: {}\n\n***\n".format(training_loss, valid_loss, valid_accuracy))
    
    file_name = save_checkpoint(model, training_dataset, args.save_dir, 'checkpoint1907')
    print("Checkpoint has been saved to {}".format(file_name))
    
    
#----------------------------------------------------------------------------------
    
if __name__ == '__main__':
    main()