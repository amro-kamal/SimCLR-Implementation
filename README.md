A pytorch-xla implementation for the paper : [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

Presentation Link: https://docs.google.com/presentation/d/14pql5IDWsfRrBS1MamidZrIcJwKP0kFk-mAytWoJVMM/edit?usp=sharing

<!-- ![simclr][logo]

[logo]: https://camo.githubusercontent.com/5ab5e0c019cdd8129b4450539231f34dc028c0cd64ba5d50db510d1ba2184160/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966 [source](https://github.com/google-research/simclr)
 -->
<img src="https://camo.githubusercontent.com/5ab5e0c019cdd8129b4450539231f34dc028c0cd64ba5d50db510d1ba2184160/68747470733a2f2f312e62702e626c6f6773706f742e636f6d2f2d2d764834504b704539596f2f586f3461324259657276492f414141414141414146704d2f766146447750584f79416f6b4143385868383532447a4f67457332324e68625877434c63424741735948512f73313630302f696d616765342e676966" alt="gif" title=[source](https://github.com/google-research/simclr) width="500"/>

[GIF SOURCE](https://github.com/google-research/simclr)

To train simclr on cofar10 use the next command:

```
python train.py     --workers=4\
                    --epochs=400\
                    --batch_size=128 
                    --projector='512-512-512'\
                    --checkpoint_dir=[pathtosaveyourmodel]
                    --seed=44\
                    --temp=0.5


```



Note that the training supports TPU only

**Results on cifar10**

|epochs   | model      |knn acc|
|---------|------------|------ |
| 400     |   Resnet18 | 82%   |
