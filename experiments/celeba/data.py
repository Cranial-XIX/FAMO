import torch, os
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class CelebaDataset(VisionDataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_dir, split='train', image_size=(64,64)):
    
        rep_file = os.path.join(data_dir, 'Eval/list_eval_partition.txt')
        self.img_dir = os.path.join(data_dir, 'Img/img_align_celeba/')
        self.ann_file = os.path.join(data_dir, 'Anno/list_attr_celeba.txt')
        self.image_size = image_size
        
        with open(rep_file) as f:
            rep = f.read()
        rep = [elt.split() for elt in rep.split('\n')]
        rep.pop()
        
        with open(self.ann_file, 'r') as f:
            data = f.read()
        data = data.split('\n')
        names = data[1].split()
        data = [elt.split() for elt in data[2:]]
        data.pop()
        
        self.img_names = []
        self.labels = []
        for k in range(len(data)):
            assert data[k][0] == rep[k][0]
            if (split=='train' and int(rep[k][1])==0) or \
                    (split=='val' and int(rep[k][1])==1) or \
                    (split=='test' and int(rep[k][1])==2):
                self.img_names.append(data[k][0])
                self.labels.append([1 if elt=='1' else 0 for elt in data[k][1:]])
        
        target_size = image_size
        self.transform = [transforms.Resize(target_size), transforms.ToTensor()]
        self.transform = transforms.Compose(self.transform)
        self.labels_rep = [[i] for i in range(40)]
        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        labels = [
            torch.tensor(self.labels[index], dtype=torch.float32)[self.labels_rep[task]] \
                    for task in range(len(self.labels_rep))
        ]
        return img, labels

    def __len__(self):
        return len(self.img_names)
