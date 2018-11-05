import torch, time, copy, sys
import matplotlib.pyplot as plt
from livelossplot import PlotLosses

def train_model_it(model, dataloaders, dataset_sizes, criterion, optimizer, batch_size, num_epochs=10, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        #Iteration
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            if scheduler != None:
                scheduler.step()
            model.train()
            running_loss = 0.0
            running_corrects = 0
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)            
            print("\rTraining Iteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders['train']), loss.item() * inputs.size(0) / batch_size ), end="")
            sys.stdout.flush()

            if (i+1) % 100 == 0:
                it_loss = running_loss / batch_size
                it_acc = running_corrects.double() / batch_size
                model.eval()
                val_loss = 0
                val_corr = 0
                for j, (inputs, labels) in enumerate(dataloaders['val']):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_corr += torch.sum(preds == labels.data)
                    print("\rValidation Iteration: {}/{}, Loss: {}.".format(j+1, len(dataloaders['val']), loss.item() * inputs.size(0) / batch_size ), end="")
                    sys.stdout.flush()                    
                valid_loss = val_loss / dataset_sizes['val']
                valid_acc  = val_corr.double() / dataset_sizes['val']

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # statistics

                liveloss.update({
                    'log loss': it_loss,
                    'val_log loss': valid_loss,
                    'accuracy': it_acc,
                    'val_accuracy': valid_acc
                })

                liveloss.draw()
                print('validation loss: {}, validation accuracy: {}'.format(valid_loss, valid_acc))
                print('Best Accuracy: {}'.format(best_acc))

                torch.save(model.state_dict(), "./models/acc_{}_loss_{}.pt".format(best_acc, valid_loss)) 
                
#         print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
#         print(  'Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
#         print('Best Val Accuracy: {}'.format(best_acc))
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model