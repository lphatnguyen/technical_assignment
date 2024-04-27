import cv2
import torchvision
import argparse
import torch
from models.lenet import LeNet
from models.resnet import resnet18, resnet34
from models.mobilenet_v2 import mobilenet_v2

# Defining arguments for inputs
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = "lenet", help = "Choose zithin this list of models: [ 'lenet', 'resnet18, 'resnet34', 'mobilenet_v2]")
parser.add_argument('--is_ferplus', type = int, default = 0)
parser.add_argument('--saving_fn', type = str, default = "best_weights", help = "Model weights file path")
parser.add_argument('--image_fn', type = str, default = "best_weights", help = "Image file path")
args = parser.parse_args()
print(args)
save_path = "results"

if args.is_ferplus:
    classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
else:
    classes = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']

def infere(model, image_fn, device):

    image = cv2.imread(image_fn)
    image = cv2.resize(image, (48, 48))
    if image.shape[2] != 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    img_tensor = torch.tensor(image, dtype = torch.float).unsqueeze(0).unsqueeze(0).to(device)
    output = model(img_tensor)
    if args.is_ferplus:
        predicted_labels = [1 if i > 0.1 else 0 for i in output.squeeze().cpu().data]
        if sum(predicted_labels) == 0:
            predicted_class = output.squeeze().argmax()
            print("The predicted emotion is", classes[predicted_class])
        else:
            predicted_classes = [classes[i] for i in range(len(classes)) if predicted_labels[i] == 1]
            print("Predicted classes are:", predicted_classes)
    else:
        predicted_label = output.argmax()
        print("Predicted label is", classes[predicted_label])

def main_infere():
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

    best_model_weights = torch.load("{}/{}.pth".format(save_path, args.saving_fn), map_location=torch.device(DEVICE))
    model.load_state_dict(best_model_weights)

    infere(model, args.image_fn, DEVICE)

main_infere()