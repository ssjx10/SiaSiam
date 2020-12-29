import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50

'ref: https://github.com/PatrickHua/SimSiam'
'author seung-wan.J'
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster.
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class BasicConv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, norm=True):
        super(BasicConv, self).__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_block=3):
        super().__init__()
        ''' 
        page 3 baseline setting
        Projection MLP.
        3 layers
        '''
        
        layers = []
        for i in range(num_block-1):
            layers.append(BasicConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(BasicConv(in_dim, out_dim, relu=False))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.proj(x)
        
        return x 


class Prediction(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048, num_block=2): # bottleneck structure
        super().__init__()
        ''' 
        page 3 baseline setting
        Prediction MLP.
        2 layers
        '''
        
        layers = []
        for i in range(num_block-1):
            layers.append(BasicConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(BasicConv(in_dim, out_dim, relu=False, norm=False))
        self.pred = nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.pred(x)
        
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50, pretrain=False):
        super().__init__()
        
        self.backbone = backbone(pretrained=pretrain)
        self.backbone.out_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        
        self.projector = Projection(self.backbone.out_features)
        # print(self.backbone)
        
        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        
        self.predictor = Prediction() # h
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss
    
    
if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)
    
    #trainset = datasets.ImageNet('datasets/ImageNet/train/', split='train',         transform=self.train_transforms, target_transform=None, download=True)
    #valset = datasets.ImageNet('datasets/ImageNet/val/', split='val', transform=self.val_transforms, target_transform=None, download=True)