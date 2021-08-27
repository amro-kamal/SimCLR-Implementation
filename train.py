import argparse
import torchvision
import torchvision.datasets as datasets 
import torch
from torch import nn, optim
import torch_xla.distributed.parallel_loader as pl
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import argparse
from pathlib import Path
import logging
from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import math
from torchsummary import summary
import pl_bolts
import torch.nn.functional as F

from src import SimCLR
from src import Transform

logger=init_logger()

#TODO : save ad load the model and optimizer                   
def main():
    '''
     args={'model':model, 'epochs':epochs, 'batch_size':batch_size, 'num_workers':num_workers,
            'optimizer':optimizer, 'transforms':Transform(), 'seed':seed}
    '''
    SEED=args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # xm.set_rng_seed(SEED)

    print('building Resnet model.....')
    model = SimCLR(args)
 
    start_epoch=0
    args.optimizer_state=None
    # logger.info(f'load model {args.load_model}')
    # if  args.load_model and (args.checkpoint_dir / 'checkpoint.pth').is_file():
    #     logger.info(f'loading the model to continue training.....')
    #     ckpt = torch.load(args.checkpoint_dir / 'resnet.pth')# map_location='cpu')
    #     start_epoch = ckpt['epoch']
    #     model.load_state_dict(ckpt['model'])
    #     args.optimizer_state=ckpt['optimizer']

    args.continue_from = start_epoch
    args.model = xmp.MpModelWrapper(model)

    args.transforms=Transform()
 
    print(f'calling spawn ...')
    xmp.spawn(XLA_trainer, args=(args,), nprocs=8, start_method='fork')

SERIAL_EXEC = xmp.MpSerialExecutor()

def XLA_trainer(index, args):
        
    device = xm.xla_device()  

    def get_data():
      train_dataset = datasets.CIFAR10(
          "/data",
          train=True,
          download=True,
          transform=args.transforms
          ) 
      return train_dataset

    train_dataset=SERIAL_EXEC.run(get_data)

    # Creates the (distributed) train sampler, which let this process only access its portion of the training dataset.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    # Creates dataloaders, which load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True)

    #TODO
    knn_train_loader, knn_val_loader  = None,None#cifar10_loader(64)

    train(args.model.to(device), args.epochs, train_loader ,device, knn_train_loader, knn_val_loader)

def train(model, epochs, train_loader, device, knn_train_loader, knn_val_loader):
    global_step=0
    writer=SummaryWriter(args.checkpoint_dir/'tensorboard')
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=10, max_epochs=800, warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)

    #for continuing training
    if args.optimizer_state:
        #The loaded optimizer must be compatible with the choice in args.optimizer
        optimizer.load_state_dict(args.optimizer)
    best_acc=0
    for epoch in range(args.continue_from, args.continue_from+epochs):
        epoch_loss=0
        num_examples=0
        epoch_start_time = time.time()
        model.train()

        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) #ParallelLoader, so each TPU core has unique batch
        # step = epoch * len(para_train_loader)-1
        for ((x1, x2), _) in para_train_loader: 
            optimizer.zero_grad()
            loss=model(x1, x2)
            if xm.is_master_ordinal:
              writer.add_scalar('ssloss',loss.item(),global_step=global_step)
              global_step+=1

            loss.backward()
            xm.optimizer_step(optimizer)
            epoch_loss += x1.shape[0] * loss.item()     #update the running loss
            num_examples+=x1.shape[0]
          
        scheduler.step()
        knn_acc =0# knn_test(model.children())[0], knn_train_loader, knn_val_loader, epoch, args)
        # if xm.is_master_ordinal:
        #     writer.add_scalar('knn acc',val_acc,global_step=epoch)

       #saving the model if the loss dicreased  
        epoch_loss=epoch_loss/num_examples
        if knn_acc >=best_acc:
          best_acc=knn_acc
          state = dict(epoch=epoch+1, best_knn_acc=best_acc, model=model.state_dict(), optimizer=optimizer.state_dict())
          # state = dict(epoch=epoch+1, best_knn_acc=best_acc, model=model.resnet_backbone.state_dict(),projector=model.projector, optimizer=optimizer.state_dict())
          xm.save(state, args.checkpoint_dir / f'checkpoint.pth' , master_only=True, global_master=False)
          xm.master_print('model saved')
        xm.master_print(f'epoch {epoch+1} ended, loss : {epoch_loss:.2f}, time: {int(time.time() - epoch_start_time)}, global steps: {global_step}')
        
    #save the final model, not necessary the best!!! 
    if xm.is_master_ordinal():
      print(f'saving the final model , loss: {epoch_loss:.2f} .....🤪🤪🤪🤪🤪🤪🤪')
      ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',map_location='cpu')
      model.load_state_dict(ckpt['model'])

    xm.save( list( model.children())[0].state_dict(), args.checkpoint_dir/'resnet.pth',master_only=True, global_master=False)
    logger.info('model saved')

args = argparse.Namespace(
                    workers=4, 
                    epochs=400,
                    batch_size=128, 
                    learning_rate_weights=0.5,
                    learning_rate_biases=0.0048,
                    weight_decay=1e-4, 
                    optimizer='SGD', 
                    projector='512-512-512',
                    checkpoint_dir=Path('gdrive/MyDrive/SimCLR/simclr_resnet18_400_lr5e1'),
                    print_model_summary = False,
                    seed=44,
                    temp=0.5
)
parser = argparse.ArgumentParser(description='Train SimCLR on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--checkpoint-dir', default='../gdrive/MyDrive/BarlowTwins/checkpoint/renet18/selfsupervised/resnet.pth')
parser.add_argument('--lr', '--learning-rate-weights', default=0.5, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--seed', default=44, type=int)
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--projector', default='512-512-512', type=str)

parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

parser.add_argument('--knn-t', default=0.5, type=float)

# utils

'''
args = parser.parse_args()  # running in command line
'''
args = parser.parse_args('')  # running in ipynb

if __name__ == '__main__':
    main()