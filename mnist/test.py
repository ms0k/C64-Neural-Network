import torch
import torch.nn as nn
import torchvision
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
import copy

image_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize([14, 14])]
)

ds_val = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "MNIST", train=False, download=True, transform=image_transform
    ),
    batch_size=1,
    shuffle=True,
)

model = torch.jit.load("mnist_trained_scr.pt").to("cpu")
model.eval()

"""qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
model_prep= prepare_fx(model, {"":qconfig}, None)
for pic, label in ds_val:
	model_prep(pic)
	break
model_quant=convert_fx(copy.deepcopy(model_prep))"""

"""model.qconfig=torch.quantization.get_default_qconfig('fbgemm')
model=torch.quantization.fuse_modules(model, [['conv1', 'relu', "conv2", "fc"]])
model=torch.quantization.prepare(model)
model(next(enumerate(ds_train))[1])
model_quant=torch.quantization.convert(model)"""
model_quant = None

# model_quant=torch.quantization.quantize_dynamic(model, None, dtype=torch.qint8)


def pic_to_c(pic, label):
    pic = pic[0][0]
    ret = "const uint8_t number_" + str(label[0].item()) + "_raw[14*14]={"
    for y in range(pic.shape[1]):
        for x in range(pic.shape[0]):
            ret += str(int(255.0 - pic[y, x].item() * 255))
            if x != pic.shape[0] - 1:
                ret += ","
        if y != pic.shape[1] - 1:
            ret += ","
    ret += "};"
    return ret


for pic, label in ds_val:
    logits_f = model(pic).detach()
    result_f = torch.argmax(logits_f, dim=1)
    print("f", result_f, label)
    print("f", logits_f)
    if model_quant:
        result_q = torch.argmax(model_quant(pic), dim=1)
        print("q", result_q, label, result_f == result_q)
    print(pic_to_c(pic, label))
    break
