import torch
import torch.optim as optim
import torch.utils
import torchvision
import cv2
import time
import os
from datasets.FerPlusDataset import FerPlusDataset, FerSingleLabel
from models.lenet import LeNet
from models.resnet import resnet18, resnet34
from models.mobilenet_v2 import mobilenet_v2
import argparse
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix
from matplotlib import pyplot as plt

# Defining arguments for inputs
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = "lenet", help = "Choose zithin this list of models: [ 'lenet', 'resnet18, 'resnet34', 'mobilenet_v2]")
parser.add_argument('--is_ferplus', type = int, default = 0)
parser.add_argument('--base_folder', type = str, default = "ferPlus2016/data", help = "Validation set needed")
parser.add_argument('--fer_path', type = str, default = "ferPlus2016/fer2013/fer2013.csv", help = "Fer csv file path")
parser.add_argument('--ferplus_path', type = str, default = "ferPlus2016/fer2013/fer2013new.csv", help = "Fer+ csv file path")
parser.add_argument('--saving_fn', type = str, default = "best_weights", help = "Fer+ csv file path")
parser.add_argument('--epochs', type = int, default = 40)
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--device', type = str, default = 'cuda', help = "[cpu, cuda]")
args = parser.parse_args()
print(args)

if args.is_ferplus:
    print("FER+ dataset is used")
    classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
else:
    print("FER dataset is used")
    classes = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']
if not os.path.exists("results/"):
    os.makedirs("results/")
save_path = "results/"

def train_epoch(model, optimizer, data_loader, loss_history, criterion, device = "cuda:0"):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.cpu().item()))
            loss_history.append(loss.cpu().item())

def evaluate(model, data_loader, loss_history, criterion, device = "cuda:0"):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    predicted_samples = []
    groundtruths = []
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            if args.is_ferplus:
                predicted_samples += [[1 if i > 0.1 else 0 for i in data] for data in output.cpu().data]
            else:
                predicted_samples += output.argmax(dim = 1).cpu().data.tolist()
            loss = criterion(output, target.to(device))

            total_loss += loss.cpu().item()
            groundtruths += target.type(torch.IntTensor).data.tolist()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + ' (' +
          '{:4.2f}'.format(100.0 * accuracy_score(groundtruths, predicted_samples)) + '%)\n')

    return accuracy_score(groundtruths, predicted_samples)

def train(model, train_loader, val_loader, optimizer, criterion, epochs = 100, device = "cuda:0"):
    train_loss_history, val_lost_history = [], []
    best_acc = 0
    for epoch in range(1, epochs + 1):
        print("Epoch {}/{}:".format(epoch, epochs))
        train_epoch(model=model, optimizer=optimizer, data_loader=train_loader, loss_history=train_loss_history, criterion=criterion)
        epoch_acc = evaluate(model=model, data_loader=val_loader, loss_history=val_lost_history, criterion=criterion)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "{}/{}.pth".format(save_path, args.saving_fn))
    return train_loss_history, val_lost_history

def train_multi_label(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()])
    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.ToTensor()])

    train_dataset = FerPlusDataset(base_folder=args.base_folder, 
                                   ferplus_path=args.ferplus_path,
                                   num_classes=len(classes),
                                   transform=train_transform)
    valid_dataset = FerPlusDataset(base_folder=args.base_folder, 
                                   ferplus_path=args.ferplus_path, 
                                   num_classes=len(classes),
                                   subset="PublicTest",
                                   transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = 8)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = 8)
    
    if args.model == 'lenet':
        model = LeNet(num_classes = len(classes), in_channels = 1).to(args.device)
    elif args.model == 'resnet18':
        model = resnet18(num_classes = 8).to(args.device)
    elif args.model =='resnet34':
        model = resnet34(num_classes = 8).to(args.device)
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(num_classes = 8).to(args.device)
    else:
        print("Not valid model, please do python train.py --help and choose again.")
        return 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loss, val_loss = train(model = model,
                                 train_loader = train_loader,
                                 val_loader = valid_loader,
                                 optimizer = optimizer,
                                 criterion = criterion,
                                 epochs = args.epochs)

def train_single_label(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()])
    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.ToTensor()])

    train_dataset = FerSingleLabel(ferPath=args.fer_path, 
                                   ferPlusPath=args.ferplus_path,
                                   subset="Training",
                                   transform=train_transform)
    valid_dataset = FerSingleLabel(ferPath=args.fer_path, 
                                   ferPlusPath=args.ferplus_path,
                                   subset="PublicTest",
                                   transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = 8)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True,
                                               num_workers = 8)
    
    if args.model == 'lenet':
        model = LeNet(num_classes = 7, in_channels = 1).to(args.device)
    elif args.model == 'resnet18':
        model = resnet18(num_classes = 7).to(args.device)
    elif args.model =='resnet34':
        model = resnet34(num_classes = 7).to(args.device)
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(num_classes = 7).to(args.device)
    else:
        print("Not valid model, please do python train.py --help and choose again.")
        return 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda:0") # to change to args.device

    train_loss, val_loss = train(model = model,
                                 train_loader = train_loader,
                                 val_loader = valid_loader,
                                 optimizer = optimizer,
                                 criterion = criterion,
                                 epochs = args.epochs)
    
    best_model_weights = torch.load("{}/{}.pth".format(save_path, args.saving_fn))
    model.load_state_dict(best_model_weights)
    print("Getting private test value")
    test_loss = []
    gt, psp = test(model = model,
                   data_loader = test_loader,
                   criterion =criterion)
    print(classification_report(gt, psp, target_names=classes))
    conf_mat = confusion_matrix(gt, psp)
    print(conf_mat)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=classes)
    disp.plot()
    plt.savefig("confmat_{}.pdf".format(args.saving_fn))

if args.is_ferplus:
    train_multi_label(args=args)
else:
    train_single_label(args=args)