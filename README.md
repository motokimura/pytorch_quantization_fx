# pytorch_quantization

With this repository, you can try model quantization of MobileNetV2 trained on CIFAR10 dataset.
Currently, post training static quantization and quantization aware training are suppored.


## Requirements

- Ubuntu OS
- CUDA (tested with 11.1)
- Python3 (test with 3.6.9)

See [requirements.txt](requirements.txt) for additional requirements.

May work with other versions, but note that torch>=1.3.0 is required to use PyTorch quantization library.


## Setup

```
$ pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

Before training, get your API key from [W&B](https://wandb.ai) and then:

```
$ echo 'WANDB_API_KEY = "xxxx"' > .env  # replace xxxx with your own W&B API key
```

`train.py` will load the API key from `.env` to send training logs to W&B.


## Post training static quantization

You need to train float model first:

```
$ EXP_ID=0
$ python train.py $EXP_ID --mode normal --lr 0.005 --batch_size 128
```

Trained weight is saved into `models/exp_0000/best_model.pth`.

To evaluate this model:

```
$ python test.py $EXP_ID --mode normal
```

You can apply post training quantization to this float model:

```
$ python test.py $EXP_ID --mode ptq --replace_relu --fuse_model
```

To compare the model size:

```
$ ls -lh models/exp_0000/scripted_*
...
-rw-r--r-- 1 kimura kimura  14M May 27 04:22 scripted_model_normal.pth  # floating
-rw-r--r-- 1 kimura kimura 3.8M May 27 04:37 scripted_model_ptq-relu-fused.pth  # quantized (post training quantization)
...
```


## Quantization aware training

For quantization aware training:

```
$ EXP_ID=1
$ python train.py $EXP_ID --mode qat --replace_relu --fuse_model --lr 0.0005 --batch_size 128
```

*Note that the learning rate is lower than the one used to train float model.*

Trained weight is saved into `models/exp_0001/best_model.pth`.

To evaluate this model:

```
$ python test.py $EXP_ID --mode qat  --replace_relu --fuse_model
```

To check the model size:

```
$ ls -lh models/exp_0001/scripted_*
...
-rw-r--r-- 1 kimura kimura 3.8M May 27 07:52 scripted_model_ptq-relu-fused.pth  # quantized (quantization aware training)
...
```


## TODOs

- [ ] Add a table to show model accuracy and performance
- [ ] Add more options for QAT (observer, etc.)
- [ ] Add models
- [ ] Use Docker
- [ ] Use PyTorch Lightning
- [ ] Finish docstring


## References

- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [(beta) Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
