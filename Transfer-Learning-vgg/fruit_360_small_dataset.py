import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------------------------------------------

'''
Config Variables
'''
path = "/home/mehul/code/Datasets/fruit-360-small/new"      #Enter the path of the folder that contains the Dataset

classes = [                                                 #Enter all the classes you wish to train your model on in the list
    'Watermelon',
    'Raspberry'
]

#-------------------------------------------------------------------------------------------------------------------------------------------------------

class fruit_360_small(Dataset):
    def __init__(self, root_dir, train=True , transform=None, target_transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.length = 0
        self.class_items =[]
        self.num_of_class = len(classes)
        self.class_labels = classes

        if (self.train):
            self.root_dir = os.path.join(self.root_dir,'train')
        else:
            self.root_dir = os.path.join(self.root_dir,'test')
        
        path = self.root_dir
        for c in classes:
            self.class_items.append(self.length)
            self.length+= len(os.listdir(os.path.join(self.root_dir,str(c)))) 
            
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        class_id = 0
        while (class_id < self.num_of_class) and (idx >= self.class_items[class_id]):
            class_id+=1
        class_id-=1
        label = classes[class_id]
        idx = idx - self.class_items[class_id]

        path = self.root_dir
        for p in os.listdir(path):
            if p==label:
                path = os.path.join(path,p)
                path = os.path.join(path, str(idx)+".jpg")
                img = read_image(path)
                img = img.type(torch.float)
                break
        
        label = class_id
        if (self.transform):
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(class_id)
        return img,label

if __name__=='__main__':
    dat = fruit_360_small(path,train = True)
    img,label = dat[100]
    print(f"Size of Dataset is {len(dat)}")
    img = img.permute(1,2,0)
    plt.imshow(img/255.0)
    plt.title(dat.class_labels[label])
    plt.show()
