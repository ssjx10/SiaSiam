import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

'author seung-wan.J'
# For ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
i_train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

i_val_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# For CheXpert
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
cheX_train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

cheX_val_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize
])

def read_filepaths(file, linear_eval):
    paths, labels = [], []
    
    data = np.genfromtxt(file, dtype=None, delimiter=',', names=True, encoding='UTF-8')
    paths = data['Path']
    labels = data['Pleural_Effusion'] # target for linear_eval
    
    if linear_eval:
        labels = np.array(labels, dtype=np.int)
        idx1 = np.where(labels == 1)[0]
        idx0 = np.where(labels == 0)[0]
        new_labels = np.concatenate([labels[idx1], labels[idx0]], 0)
        new_paths = np.concatenate([paths[idx1], paths[idx0]], 0)
        
        return new_paths, new_labels
            
    return paths, labels

class CheXpert(Dataset):
    def __init__(self, mode, dataset_path='data/', linear_eval=False):
        
        self.linear_eval = linear_eval
        if mode == 'train':
            datafile = 'data/CheXpert-v1.0-small/train.csv'
            self.transform = cheX_train_transform
        elif mode == 'valid':
            datafile = 'data/CheXpert-v1.0-small/valid.csv'
            self.transform = cheX_val_transform
        
        paths, labels = read_filepaths(datafile, linear_eval)
        self.paths = paths
        self.labels = np.array(labels, dtype=np.int)
            
        self.root = dataset_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        
        label = self.labels[index]
        if not self.linear_eval:
            image1, image2 = self.load_image(self.root + self.paths[index])
            return image1, image2
        else:
            image = self.load_image(self.root + self.paths[index])
            return image, label
    
    def load_image(self, img_path, dim=(320,320)):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)
        image_tensor1 = self.transform(image)
        
        if not self.linear_eval:
            image_tensor2 = self.transform(image)
            return image_tensor1, image_tensor2
        else:
            return image_tensor1
    
    
if __name__ == "__main__":
    data = CheXpert(mode='train')
    print(len(data), data[0][0].shape, data[0][1].shape)