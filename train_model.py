import copy
import os
import sys
import time
import torch
from livelossplot import PlotLosses


def train_model(output_path, model, dataloaders, criterion, optimizer, num_epochs=5, scheduler=None):
    if not os.path.exists("models/" + str(output_path)):
        os.makedirs("models/" + str(output_path))
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
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

                print(
                    "\rIteration: {}/{}, Loss: {}.".format(
                        i + 1, len(dataloaders[phase]), loss.item() * inputs.size(0)
                    ),
                    end="",
                )

                #                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            if phase == "train":
                prefix = ''
                # avg_loss = epoch_loss
                # t_acc = epoch_acc
            else:
                prefix = 'val_'
                # val_loss = epoch_loss
                # val_acc = epoch_acc

            #             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()

        liveloss.update(logs)
        liveloss.send()

        torch.save(model.state_dict(), f"./models/{output_path}/model_{epoch + 1}_epoch.pt")
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best))
