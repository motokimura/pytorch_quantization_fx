# https://github.com/pytorch/vision/blob/master/torchvision/models/quantization/mobilenetv2.py

import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub, fuse_modules
from torchvision.models.mobilenetv2 import (ConvBNReLU, InvertedResidual,
                                            MobileNetV2, model_urls)
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['QuantizableMobileNetV2', 'mobilenet_v2']


def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        # Replace torch.add (of floating mobilenet_v2)
        # with FloatFunctional to make the model quantizable
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        """MobileNet V2 main class

        Args:
           Inherits args from floating point MobileNetV2
        """
        super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()


# TODO: update docstring
def mobilenet_v2(pretrained=None,
                 progress=True,
                 replace_relu=False,
                 fuse_model=False,
                 **kwargs):
    """Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (str or None):
        progress (bool): If True, displays a progress bar of the download to stderr
        replace_relu (bool):
        fuse_model (bool):

    Returns:
        QuantizableMobileNetV2: model in eval mode.
    """
    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)

    if pretrained is not None:
        if pretrained == 'imagenet':
            # load imagenet pretrained mobilenet_v2 (floating weight)
            model_url = model_urls['mobilenet_v2']
            state_dict = load_state_dict_from_url(model_url, progress=progress)
        else:
            state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)

    if replace_relu:
        # replace ReLU6 with ReLU
        # so that we can "fuse" Conv+BN+ReLU modules later
        _replace_relu(model)

    if fuse_model:
        # fuse Conv+BN and Conv+BN+ReLU modules prior to quantization
        # this operation does not change the numerics
        # this can both make the model faster by saving on memory access while also improving numerical accuracy
        # while this can be used with any model, this is especially common with quantized models
        assert replace_relu, '`replace_relu` must be True if you want to fuse modules.'
        # XXX: for some reason we don't know, convert for post-training quantization fails w/o this eval...
        model.eval()
        # fuse Conv+BN and Conv+BN+ReLU modules
        model.fuse_model()

    return model.eval()
