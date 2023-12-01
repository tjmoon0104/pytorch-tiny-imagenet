import time

import torch


def test_model(model, dataloaders, criterion, device):
    since = time.time()
    phase = "test"

    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

    time_elapsed = time.time() - since
    print("Test Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    print("Test complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
