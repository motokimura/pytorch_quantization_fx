# pytorch_quantization

With this repository, you can try model quantization of MobileNetV2 trained on CIFAR10 dataset.
Currently, post training static quantization and quantization aware training are suppored.

|model               |quantization method                |CIFAR10 val accuracy [%] |model size [MB]
|---                 |---                                |---                      |---
|MobileNetV2 (float) |-                                  |96.36                    |14
|MobileNetV2 (int8)  |post training static quantization  |95.53                    |3.8
|MobileNetV2 (int8)  |quantization aware training        |96.30                    |3.8

## Requirements

- Ubuntu OS
- CUDA (tested with 11.6)
- Python3 (test with 3.8.8)

See [requirements.txt](requirements.txt) for additional requirements.

May work with other versions, but note that torch>=1.3.0 is required to use PyTorch quantization library.

## Setup

```
$ pip install -r requirements.txt
```

Before training, sign up for [W&B](https://wandb.ai)
and create a new project named `pytorch_model_quantization`.

Get your API key from [W&B](https://wandb.ai) > `Settings` > `API keys` and then:

```
$ echo 'WANDB_API_KEY = "xxxx"' > .env  # replace xxxx with your own W&B API key
```

`train.py` will load the API key from `.env` to send training logs to W&B.

## Pretrained weights

Pretrained weights are available:

```
unzip models_v2.zip
```

- `models/exp_2000/model_best.pth`: float model
- `models/exp_2001/model_best.pth`: model trained with qnantization-aware training

## Post training static quantization

You need to train float model first (can be skipped if you use pretrained weight):

```
$ EXP_ID=0
$ python train.py $EXP_ID --mode normal --lr 0.005 --batch_size 64
```

Trained weight is saved into `models/exp_2000/best_model.pth`.

To evaluate this model:

```
$ python test.py $EXP_ID --mode normal
```

You can apply post training static quantization to this float model:

```
$ python test.py $EXP_ID --mode ptq --replace_relu --fuse_model
```

To compare the model size:

```
$ ls -lh models/exp_2000/scripted_*
...
-rw-r--r-- 1 kimura kimura  14M May 27 04:22 scripted_model_normal.pth  # floating
-rw-r--r-- 1 kimura kimura 3.8M May 27 04:37 scripted_model_ptq-relu-fused.pth  # quantized (post training static quantization)
...
```

## Quantization aware training

For quantization aware training (can be skipped if you use pretrained weight):

```
$ EXP_ID=1
$ python train.py $EXP_ID --mode qat --replace_relu --fuse_model --lr 0.005 --batch_size 64
```

Trained weight is saved into `models/exp_2001/best_model.pth`.

To evaluate this model:

```
$ python test.py $EXP_ID --mode qat  --replace_relu --fuse_model
```

To check the model size:

```
$ ls -lh models/exp_2001/scripted_*
...
-rw-r--r-- 1 kimura kimura 3.8M May 27 07:52 scripted_model_ptq-relu-fused.pth  # quantized (quantization aware training)
...
```

## TODOs

- [x] Add a table to show model accuracy and performance
- [ ] Add more options for QAT (observer, etc.)
- [ ] Add models
- [ ] Finish docstring

## References

- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [(beta) Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [torchvision/references/classification#quantized](https://github.com/pytorch/vision/tree/main/references/classification#quantized)
