# SimCLR Self-Supervised Learning/Contrastive Learning

# Data: CIFAR10
Encoder: ResNet18, ResNet34, ResNet50
Projection Head: 2-Layers MLP (ResNet18, ResNet34), 3-Layers MLP(ResNet50)
Decoder: Fully connection Layer

# Acc:

ResNet18 based: 

72.52% (100 epochs).

86.55% (200 epochs).

The loss is still decreasing. Try more training epochs for proxy task, the performance will be improved.

ResNet34 based: Have not tested yet, but code is ready.

ResNet50 based: Have not tested yet, but code is ready.

# Loss:

lighting NT-Xent Loss

# Optimizer:

SGD (You can try LARS if you have a good GPU, try large batch size)

# Scheduler:

Cosine Annealing

# Unsupervised Training
<p float="left">
  <img src="/average_image_loss_vs_learning_rate.png" width="600" />
  <img src="/batch_loss_vs_learning_rate.png" width="600" />
  <img src="/unsupervised_learning_avg_batch_loss_vs_epoch.png" width="600" /> 
</p>

# Downstream Training

<p float="left">
  <img src="/Down_stream_batch_loss_vs_learning_rate.png" width="600" />
  <img src="/Down_stream_imageloss_vs_learning_rate.png" width="600" />
  <img src="/Downstream_batch_loss_vs_epoch.png" width="600" /> 
</p>

# If you reference the code, please add the link. Thanks
