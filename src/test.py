from datasets.FerPlusDataset import FerPlusDataset, FerSingleLabel
from models.lenet import LeNet
from models.resnet import resnet18, resnet34
from models.mobilenet_v2 import mobilenet_v2
import argparse
import numpy as np
import torch
import torchvision
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
parser.add_argument('--batch_size', type = int, default = 256)
args = parser.parse_args()
print(args)
save_path = "results/"

if args.is_ferplus:
    print("FER+ dataset is used")
    classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
else:
    print("FER dataset is used")
    classes = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']

def test(model, data_loader, criterion, device = "cuda:0"):
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
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + ' (' +
          '{:4.2f}'.format(100.0 * accuracy_score(groundtruths, predicted_samples)) + '%)\n')

    return groundtruths, predicted_samples

def main_test():
    use_gpu = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_gpu else 'cpu')
    if args.is_ferplus:
        if args.model == 'lenet':
            model = LeNet(num_classes = len(classes), in_channels = 1).to(DEVICE)
        elif args.model == 'resnet18':
            model = resnet18(num_classes = 8).to(DEVICE)
        elif args.model =='resnet34':
            model = resnet34(num_classes = 8).to(DEVICE)
        elif args.model == 'mobilenet_v2':
            model = mobilenet_v2(num_classes = 8).to(DEVICE)
        else:
            print("Not valid model, please do python train.py --help and choose again.")
            return
    else:
        if args.model == 'lenet':
            model = LeNet(num_classes = 7, in_channels = 1).to(DEVICE)
        elif args.model == 'resnet18':
            model = resnet18(num_classes = 7).to(DEVICE)
        elif args.model =='resnet34':
            model = resnet34(num_classes = 7).to(DEVICE)
        elif args.model == 'mobilenet_v2':
            model = mobilenet_v2(num_classes = 7).to(DEVICE)
        else:
            print("Not valid model, please do python train.py --help and choose again.")
            return

    if args.is_ferplus:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.ToTensor()])

    if args.is_ferplus:
        test_dataset = FerPlusDataset(base_folder=args.base_folder,
                                    ferplus_path=args.ferplus_path,
                                    num_classes=len(classes),
                                    subset="PrivateTest",
                                    transform=transform)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size = args.batch_size, 
                                                shuffle = True, 
                                                num_workers = 8)
    else:
        test_dataset = FerSingleLabel(ferPath=args.fer_path, 
                                      ferPlusPath=args.ferplus_path,
                                      subset="PrivateTest",
                                      transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size = args.batch_size, 
                                                  shuffle = True, 
                                                  num_workers = 8)
    
    best_model_weights = torch.load("{}/{}.pth".format(save_path, args.saving_fn), map_location=torch.device(DEVICE))
    model.load_state_dict(best_model_weights)
    
    groundtruths, predicted_samples = test(model = model,
                                           data_loader = test_loader,
                                           criterion =criterion,
                                           device = DEVICE)
    
    print(" Classification report:")
    print(classification_report(groundtruths, predicted_samples, target_names=classes))
    if args.is_ferplus:
        conf_mat = multilabel_confusion_matrix(groundtruths, predicted_samples)
        plot_multi_label_classification(np.array(groundtruths), np.array(predicted_samples), classes)
        plt.savefig("confmat_{}.pdf".format(args.saving_fn))
    else:
        conf_mat = confusion_matrix(groundtruths, predicted_samples)
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=classes)
        disp.plot()
        plt.savefig("confmat_{}.pdf".format(args.saving_fn))

def plot_multi_label_classification(groundtruths, predicted_samples, classes):
    f, axes = plt.subplots(2, len(classes)//2, figsize=(25, 15))
    axes = axes.ravel()
    for i in range(len(classes)):
        disp = ConfusionMatrixDisplay(confusion_matrix(groundtruths[:, i],
                                                       predicted_samples[:, i]),
                                      display_labels=[0, classes[i]])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'class {classes[i]}')
        if i<10:
            disp.ax_.set_xlabel('')
        if i%5!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    f.colorbar(disp.im_, ax=axes)

main_test()