
import os
import argparse
import mlflow
import torchvision
from torchvision import transforms
import torch

def train_model(model,criterion,optimizer,loader,n_epochs,device):
    
    loss_over_time = [] # to track the loss as the network trains
    
    model = model.to(device) # Send model to GPU if available
    model.train() # Set the model to training mode
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        running_corrects = 0
        
        for i, data in enumerate(loader):
            # Get the input images and labels, and send to GPU if available
            inputs, labels = data[0].to(device), data[1].to(device)
            # Convert to one channel image (grayscale)
            inputs = inputs[:,0,:,:].unsqueeze(1)

            # Zero the weight gradients
            optimizer.zero_grad()

            # Forward pass to get outputs
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation to get the gradients with respect to each weight
            loss.backward()

            # Update the weights
            optimizer.step()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item()

            # Convert loss into a scalar and add it to running_loss
            running_loss += loss.item() * inputs.size(0)
            # Track number of correct predictions
            running_corrects += torch.sum(preds == labels.data)
            
        # Calculate and display average loss and accuracy for the epoch
        epoch_loss = running_loss / len(loader)
        epoch_acc = running_corrects.double() / len(loader)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        loss_over_time.append(epoch_loss)

    return loss_over_time

def main():
    """Main function of the script."""
    print('Starting...')
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # start Logging
    mlflow.start_run()

    ###################
    #<prepare the data>
    ###################
    train_path = os.path.join(args.data, 'train')
    test_path = os.path.join(args.data, 'test')
    
    # Create datasets    
    train_data = torchvision.datasets.ImageFolder(train_path, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])]))
    test_data = torchvision.datasets.ImageFolder(test_path, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])]))
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    ###################
    #</prepare the data>
    ###################

    ##################
    #<train the model>
    ##################
    
    model = torchvision.models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the resnet input layer to take in grayscale images (1 input channel), since it was originally trained on color (3 input channels)
    in_channels = 1
    model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace the resnet final layer with a new fully connected Linear layer we will train on our task
    # Number of out units is number of classes (3)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train the model
    n_epochs = 10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    cost_path_train = train_model(model,criterion,optimizer,train_dataloader,n_epochs,device)
    
    # Test the model
    cost_path_test = train_model(model,criterion,optimizer,test_dataloader,n_epochs,device)
    
    # Print the train and test loss
    print(f"Train loss: {cost_path_train[-1]}")
    print(f"Test loss: {cost_path_test[-1]}")
    
    ##################
    #</train the model> 
    ##################

    ##########################
    #<save and register model>
    ##########################
    # registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.pytorch.log_model(model, args.registered_model_name)

    # saving the model to a file
    mlflow.pytorch.save_model(model, path='latest_model')
    
    # stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
