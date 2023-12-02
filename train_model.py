import time
from pathlib import Path

import torch
from livelossplot import PlotLosses


def train_model(output_path, model, dataloaders, criterion, optimizer, device, num_epochs=5, scheduler=None) -> int:
    (Path("models") / output_path).mkdir(parents=True, exist_ok=True)
    since = time.time()
    liveloss = PlotLosses()

    best_acc = 0.0
    best = 0

    for epoch in range(num_epochs):
        logs = {}
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            if phase == "train":
                prefix = ""
            else:
                prefix = "val_"

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1

            logs[prefix + "log loss"] = epoch_loss.item()
            logs[prefix + "accuracy"] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.send()

        torch.save(model.state_dict(), f"./models/{output_path}/model_{epoch + 1}_epoch.pt")
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best))
    return best
