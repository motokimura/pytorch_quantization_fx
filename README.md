# pytorch_quantization_fx

With this repository, you can try model quantization of MobileNetV2 trained on CIFAR-10 dataset.
Currently, only post training static quantization is supported.

|model               |quantization method                |CIFAR-10 test accuracy [%] |model size [MB]
|---                 |---                                |---                      |---
|MobileNetV2 (float) |-                                  |96.61                    |14
|MobileNetV2 (int8)  |post training static quantization  |96.24                    |3.8

## Setup

```
docker compose build dev
docker compose run dev bash
```

Before training, sign up for [W&B](https://wandb.ai)
and create a new project named `pytorch_quantization_fx`.

Get your API key from [W&B](https://wandb.ai) > `Settings` > `API keys` and then:

```
echo 'WANDB_API_KEY = "xxxx"' > .env  # replace xxxx with your own W&B API key
```

When you run `tools/train.py`, the API key will be loaded from `.env` to send training logs to W&B.

## Pretrained weights

Pretrained weights are available:

```
unzip models_v3.zip
```

```
models
└── exp_3000
    ├── 20230207_03:59:54.json
    └── best_model.pth  # float model
```

## Post training static quantization

Train float model first (can be skipped if you use the pretrained weight above):

```
EXP_ID=3000
python tools/train.py $EXP_ID
```

Trained weight is saved into `models/exp_3000/best_model.pth`.

To evaluate this model:

```
python tools/test.py $EXP_ID --mode float
```

Apply post training static quantization to this float model:

```
python tools/test.py $EXP_ID --mode ptq
```

To compare the model size:

```
ls -lh models/exp_3000/scripted_*

...
-rw-r--r-- 1 root root  14M Feb  8 00:01 models/exp_3000/scripted_model_float.pth  # float
-rw-r--r-- 1 root root 3.8M Feb  7 23:58 models/exp_3000/scripted_model_ptq.pth  # quantized (with post training static quantization)
...
```

## TODOs

- [ ] Sensitivity analysis
- [ ] Quantization aware training

## References

- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)
- [(PROTOTYPE) FX GRAPH MODE POST TRAINING STATIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
- [PYTORCH FX NUMERIC SUITE CORE APIS TUTORIAL](https://pytorch.org/tutorials/prototype/fx_numeric_suite_tutorial.html)
- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [(beta) Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [torchvision/references/classification#quantized](https://github.com/pytorch/vision/tree/main/references/classification#quantized)
