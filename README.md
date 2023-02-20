# pytorch_quantization_fx

With this repository, you can try model quantization of MobileNetV2 trained on CIFAR-10 dataset with PyTorch.
Post training static quantization (PTQ) and quantization aware training (QAT) are supported.

|model               |quantization method                |CIFAR-10 test accuracy [%] |model size [MB]
|---                 |---                                |---                      |---
|MobileNetV2 (float) |-                                  |96.43                    |14
|MobileNetV2 (int8)  |post training static quantization  |95.99                    |3.8
|MobileNetV2 (int8)  |quantization aware training        |96.48                    |3.8

## Setup

```
docker compose build dev
docker compose run dev bash
```

Before training, sign up for [W&B](https://wandb.ai)
and create a new project named `pytorch_quantization_fx`.

Get your API key from [W&B](https://wandb.ai) > `Settings` > `API keys` and then:

```
echo 'WANDB_API_KEY = xxxx' > .env  # replace xxxx with your own W&B API key
```

When you run `tools/train.py`, the API key will be loaded from `.env` to login W&B.

If `tools/train.py` failed to send logs to W&B, run following and re-run `tools/train.py`.

```
git config --global --add safe.directory /work
```

## Pretrained weights

Pretrained weights are available:

```
unzip models_v4.zip
```

```
models
├── exp_4000
│   ├── 20230210_11:20:36.json
│   └── best_model.pth  # float model
└── exp_4001
    ├── 20230212_08:38:59.json
    └── best_model.pth  # float model trained with quantization aware training
```

## Post training static quantization

Train float model first (can be skipped if you use the pretrained weight above):

```
EXP_ID=4000
python tools/train.py $EXP_ID
```

Trained weight is saved into `models/exp_4000/best_model.pth`.

To evaluate this model:

```
python tools/test.py $EXP_ID
```

Apply post training static quantization to this float model:

```
python tools/test.py $EXP_ID --ptq
```

To compare the model size:

```
ls -lh models/exp_4000/scripted_*

...
-rw-r--r-- 1 1002 1002  14M Feb 12 08:04 models/exp_4000/scripted_model_float.pth  # float
-rw-r--r-- 1 1002 1002 3.8M Feb 13 02:51 models/exp_4000/scripted_model_ptq.pth  # quantized (with post training static quantization)
...
```

## Quantization aware training

Train model with quntization aware training (can be skipped if you use the pretrained weight above):

```
EXP_ID=4001
python tools/train.py $EXP_ID --qat
```

Trained weight is saved into `models/exp_4001/best_model.pth`.

To evaluate this model:

```
python tools/test.py $EXP_ID --qat
```

To check the model size:

```
ls -lh models/exp_4001/scripted_*

...
-rw-r--r-- 1 root root 3.8M Feb 13 02:49 models/exp_4001/scripted_model_qat.pth  # quantized (with quantization aware training)
...
```

## TODOs

- [ ] Sensitivity analysis
- [x] Quantization aware training

## References

- [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)
- [(PROTOTYPE) FX GRAPH MODE POST TRAINING STATIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
- [PYTORCH FX NUMERIC SUITE CORE APIS TUTORIAL](https://pytorch.org/tutorials/prototype/fx_numeric_suite_tutorial.html)
- [github.com/fbsamples/pytorch-quantization-workshop](https://github.com/fbsamples/pytorch-quantization-workshop)
- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [(beta) Static Quantization with Eager Mode in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [torchvision/references/classification#quantized](https://github.com/pytorch/vision/tree/main/references/classification#quantized)
