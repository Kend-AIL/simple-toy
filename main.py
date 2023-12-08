import torch
from torch import nn
from model import Model
loss=nn.CrossEntropyLoss()
def train(model, optimizer, dataloader, loss_function):
    model.train()  # set the model to training mode
    total_loss = 0

    for batch in dataloader:
        inputs = batch["input"]
        gt = batch["gt"]

        # Forward pass: compute the model output
        output = model(inputs,gt[:,:-1])

        # Compute the loss
        train_loss = loss_function(output, gt[:,1:])

        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()  # clear previous gradients
        train_loss.backward()  # backpropagation

        # Update parameters
        optimizer.step()

        total_loss += train_loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss