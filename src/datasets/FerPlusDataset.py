import PIL.Image as Image
import torch
import os
from glob import glob
from itertools import islice
import csv
import sys
import numpy as np

# List of folders for training, validation and test.
folder_names = {'Training'   : 'FER2013Train',
                'PublicTest' : 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}

def str_to_image(image_blob):
    ''' Convert a string blob to an image object. '''
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    return Image.fromarray(image_data)

class FerPlusDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, ferplus_path, num_classes, subset = "Training", transform = None):
        assert subset in folder_names, "subset = {} not in this list {}".format(subset, folder_names.keys())
        self.ferplus_path = ferplus_path
        self.transform = transform
        self.num_classes = num_classes
        self.folder_path = os.path.join(base_folder, folder_names[subset])
        self.fns = glob("{}/*.png".format(self.folder_path))
        self.process_emotion_distribution()
        return
    
    def process_emotion_distribution(self):
        self.data = []
        with open(self.ferplus_path,'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                if len(row[1].strip()) > 0 and os.path.exists(os.path.join(self.folder_path, row[1].strip())):
                    emotion_values = list(map(float, row[2:len(row)]))
                    processed_emotion = self.process_emotion(emotion_values)
                    if np.argmax(processed_emotion) < self.num_classes:
                        processed_emotion = processed_emotion[:-2]
                        processed_emotion = [float(i)/sum(processed_emotion) for i in processed_emotion]
                        self.data.append((os.path.join(self.folder_path, row[1].strip()), processed_emotion))
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        fn, labels = self.data[index]
        image = Image.open(fn)
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(labels)
        return image, target
    
    def process_emotion(self, emotion_raw):
        """
        This method depends on the provided method in the FERPlus class by reimplementing the _process_data with the cross entropy mode.
        https://github.com/microsoft/FERPlus/blob/ae2128abf776409c93e50cf7c9d87180673314e6/src/ferplus.py#L203
        """  
        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0
 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        sum_part = 0
        count = 0
        valid_emotion = True
        while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
            maxval = max(emotion_raw) 
            for i in range(size): 
                if emotion_raw[i] == maxval: 
                    emotion[i] = maxval
                    emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    if i >= 8: 
                        valid_emotion = False
                        if sum(emotion) > maxval:
                            emotion[i] = 0
                            count -= 1
                        break
        if sum(emotion) <= 0.5*sum_list or count > 3:
            emotion = emotion_unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]

class FerSingleLabel(torch.utils.data.Dataset):
    def __init__(self, ferPath, ferPlusPath, subset, transform = None):
        super(FerSingleLabel, self).__init__()
        assert subset in folder_names, "subset = {} not in this list {}".format(subset, folder_names.keys())
        self.ferPath = ferPath
        self.ferPlusPath = ferPlusPath
        self.transform = transform
        self.imgs = []
        self.labels = []
        ferplus_entries = []
        with open(ferPlusPath,'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                ferplus_entries.append(row)

        index = 0
        with open(ferPath,'r') as csvfile:
            fer_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(fer_rows, 1, None):
                ferplus_row = ferplus_entries[index]
                file_name = ferplus_row[1].strip()
                if row[2] == subset and len(file_name) > 0:
                    image = str_to_image(row[1])               
                    self.imgs.append(image)
                    self.labels.append(int(row[0]))
                index += 1
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label)
        return img, label

if __name__ == "__main__":
    dataset = FerSingleLabel("ferPlus2016/fer2013/fer2013.csv", "ferPlus2016/fer2013/fer2013.csv", subset = "Training")
    print(dataset.__getitem__(10))