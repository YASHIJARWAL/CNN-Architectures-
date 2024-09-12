
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as transform 
from torchvision import datasets
from torchvision import models
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
transforms={
    'train':transform.Compose([
    transform.RandomResizedCrop(224),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
    'val':transform.Compose([
    transform.RandomResizedCrop(224),
    transform.RandomHorizontalFlip(),
    transform.ToTensor(),
    transform.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}


data_dir='flowers'
input_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),transforms[x]) for x in ['train','val']}
input_datasets
dataloaders={x:torch.utils.data.DataLoader(input_datasets[x],batch_size=4,shuffle=True,num_workers=4)for x in ('train','val')}
data_sizes={x:len(input_datasets[x])for x in ('train','val')}
class_names=input_datasets['train'].classes
class_names
model=models.resnet18(pretrained=True)
for name , param in model.named_parameters():
    if 'fc' in name:
        param.requires_grid=True
    else:
        param.requires_grid=False
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
import time

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    epoch_start = time.time()

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        phase_start = time.time()

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_sizes[phase]
        epoch_acc = running_corrects.double() / data_sizes[phase]

        phase_time = time.time() - phase_start
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {phase_time:.2f}s')

    epoch_time = time.time() - epoch_start
    print(f'Epoch {epoch+1} complete in {epoch_time:.2f}s')

print('Training complete')
torch.save(model.state_dict(), 'flower_classification_model.pth')
