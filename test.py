# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:35:02 2025

@author: Haliz369
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, classification_report)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from time import time

#%% GLOBAL VARIABLES
windowSize = 9

#%% Functions
# 
from Image_Preproc import ExtractPatches, ScaleData

#%% Load Data

# # Open a multi-band TIFF image Load the MATLAB file
mat = loadmat('Indian_pines_corrected.mat')

image_array = mat['indian_pines_corrected']
data = image_array.astype(np.float16)

from einops import rearrange

# Change the order of dimensions using einops
# Here, we're swapping the first and second dimensions
data = rearrange(data, 'i j k -> k i j')


# # Open a gt
mat = loadmat('Indian_pines_gt.mat')
gt = mat['indian_pines_gt']
print(f'Data Shape: {data.shape[1:3]}\nNumber of Bands: {data.shape[0]}')



#%%"""# Pre-processing
# This step includes:

#   1) Extract patches of the image

#   2) Encode labels of each class

#   3) Split samples into training, validation and test samples

#   4) Scaling samples to [0,1] interval



# ################# Extract patches of the the image
X, labels, pos_in_image = ExtractPatches(data,GT=gt, windowSize=windowSize)

################# Encode labels of each class

enc = OneHotEncoder()
y=enc.fit_transform(labels.reshape(-1, 1)).toarray() #turn labels to categorical (i.e., each label is represented by a vector) usig  OneHotEncoder method

################# Split samples into: 1) Training, 2) Validation and 3) Test samples

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.2,
                                                    random_state=39)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    stratify=y_train,
                                                    train_size=0.8,
                                                    random_state=39)


# ################ Scaling

# Find the minimum and maximum values of the array
min_value = np.min(X_train)
max_value = np.max(X_train)

# Scale the array between 0 and 1
X_train=ScaleData(X_train,min_value,max_value)
X_val=ScaleData(X_val,min_value,max_value)
X_test=ScaleData(X_test,min_value,max_value)


print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_validation: {X_val.shape}\ny_validation: {y_val.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")

#%% """# Train Model"""

########### Hyperparameters

input_shape = X_train[0].shape                       # input shape of  model  e.g.(200,21,21)
nb_classes= y_train.shape[1]                         # output shape of model

batch_size=64

################################################

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device,)
y_train = torch.tensor(y_train, dtype=torch.float).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float).to(device)

# Create DataLoader for training, validation, and test sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#%% Initialize the model and loss function


from ConViT_model2 import ConViT


model = ConViT(image_dim= input_shape[0], num_dense_layers= 2, growth_rate=16, denseblk_dropout_rate=0.5 ,
                  patch_size=input_shape[1], emb_dim=256,
                  num_encod_layers=4, num_heads=8, MLP_dim=256, dropkey_rate=0.2,  att_drop_rate=0.5,
                  classifier_drop_rate=0.5, num_classes= nb_classes )#.to(device) # Move the model and data to the GPU if available


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(),lr= 0.0005, weight_decay=1e-4)  # lr= 0.0005, weight_decay=1e-4

from torchinfo import summary
input = (1,) + input_shape

print(summary(model, input))

model = model.to(device) 


#%% Train the model
from tqdm import tqdm
from torch.optim import lr_scheduler

# Define the learning rate scheduler

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Lists to store training and validation loss values for each epoch
train_loss_values = []
val_loss_values = []

nb_epoch=100
best_val_loss = float('inf')
# best_val_accuracy = 0

for epoch in range(nb_epoch):
    start_time = time()
    
    model.train()
    train_loss = 0

    for inputs, labels in train_loader :  #tqdm(train_loader)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    # Step the learning rate scheduler
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.max(1)[1]).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / total * 100
    
    epoch_time= time()-start_time 
    print(f'Epoch [{epoch + 1}/{nb_epoch}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}, Time: {epoch_time:.2f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')


    # Append the training and validation loss values for plotting
    train_loss_values.append(train_loss / len(train_loader))
    val_loss_values.append(val_loss)



#%% plot Loss
import matplotlib.pyplot as plt
# Plot the loss values
epochs = np.arange(1, nb_epoch + 1)
plt.plot(epochs, train_loss_values, label='Training Loss')
plt.plot(epochs, val_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


#%% Validate the performance of the model
chunk_size = 200000  # Adjust this based on your available memory

# Define a generator function to yield data chunks
def data_chunk_generator(X_data, y_data, chunk_size):
    num_samples = X_data.shape[0]
    for start in range(0, num_samples, chunk_size):
        end = start + chunk_size
        yield X_data[start:end,:], y_data[start:end,:]



model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Create a loop to process chunks of data and save results
all_prediction=[]

for chunk in tqdm(data_chunk_generator(X_test, y_test, chunk_size), total= (X_test.shape[0]//chunk_size), desc="Processing Chunks"):

  X = torch.tensor(chunk[0], dtype=torch.float32).to(device)
  y = torch.tensor(chunk[1], dtype=torch.float).to(device)
  test_dataset = TensorDataset(X, y)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  ########## Results on test samples
  chunk_prediction=[]
  with torch.no_grad():
      for inputs in test_loader:
          outputs = model(inputs[0])
          _, predicted = outputs.max(1)
          chunk_prediction.append(predicted.cpu().numpy())

  all_prediction.append(chunk_prediction)
   # # Save or process predictions for the current chunk as needed

flattened_arrays = [np.concatenate(arr) for arr in all_prediction]
# Concatenate all the flattened arrays
result_array = np.concatenate(flattened_arrays)

 
label= np.argmax(y_test,1)

# Classification Report
print(classification_report(result_array, label))

from sklearn.metrics import *
Kappa=cohen_kappa_score(label, result_array) # calculate Kappa coefficient
OA=accuracy_score(label, result_array) # calculate Overall accuracy
matrix = confusion_matrix(label, result_array) # calculate confusion matrix
Producer_accuracy= matrix.diagonal()/matrix.sum(axis=1) # calculate producer accuracy

print(f"Overall Accuracy: {OA*100:.2f}\nKappa coefficient: {Kappa:.4f}\nMean PA: {np.mean(Producer_accuracy*100):.2f}\nProducer accuracy: {Producer_accuracy*100} ")

