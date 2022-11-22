# SimCLR Self-Supervised Learning/Contrastive Learning

# Data: CIFAR10
Encoder: ResNet18, ResNet34, ResNet50
Projection Head: 2-Layers MLP (ResNet18, ResNet34), 3-Layers MLP(ResNet50)
Decoder: Fully connection Layer

Acc: 
ResNet18: 72.52% (100 epochs), 86.55 (200 epochs), The loss is still decreasing. Try more traning epochs for proxy task, the performance will be improved.
ResNet34: Have not tested yet, but code is ready.
ResNet50: Have not tested yet, but code is ready.

Loss:
lighting NT-Xent Loss

Optimizer:
SGD (You can try LARS if you have a good GPU, try large batch size)

Scheduler:
Cosine Annealing

The code will be uploaded in Mid-December
