import torch
import torchvision
import torch.nn as nn
import copy
# import torch_xla.core.xla_model as xm

class SimCLR(nn.Module):
    ''' 
      resnet with custom FC head
    '''
    def __init__(self, args  ):
        super().__init__()
        self.args=args
        self.resnet_backbone = resnet50() #torchvision.models.resnet18(zero_init_residual=True)
        self.resnet_backbone.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(512,512),
                                                nn.ReLU(),
                                                nn.Linear(512,64))


        self.CEL = nn.CrossEntropyLoss(reduction="mean")
    def forward(self, x1, x2=None):
      if x2==None:
        return self.resnet_backbone(x1)

      else:
        z1 = self.projector(self.resnet_backbone(x1)) 
        z2 = self.projector(self.resnet_backbone(x2))

        loss = self.info_nce_loss(z1, z2, self.args.temp,self.CEL)

        return loss


    def info_nce_loss(self , z1, z2, temp , CEL):
        batch_size = z1.shape[0]
        device = z1.device
        z1 = nn.functional.normalize(z1, dim=1) #[b x d]
        z2 = nn.functional.normalize(z2, dim=1) #[b x d]
        z = torch.cat((z1, z2), axis=0) #[2*b x d]
        sim = torch.matmul(z, z.T) / temp  #[2*b x 2*b]
        # print('sim ',sim.shape)

        #We need to removed the similarities of samples to themselves
        off_diag_ids = ~torch.eye(2*batch_size, dtype=torch.bool, device=device)
        logits = sim[off_diag_ids].view(2*batch_size, -1)  #[2*b x 2*b-1]
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels])

        loss = CEL(logits, labels)
        return loss