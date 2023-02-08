"""
MNist classifier trainer, using "MNIST for AVR example" in the onnx2c repository as reference.
Network architecture hyperparameters are also adapted from there.
I could not reproduce the results with the original neural network code, so this is my rewrite with a few optimizations:
1. Training speed: The CPU is mostly used instead of the GPU despite occupying VRAM which I assume is overhead due to small network size.
With the training setup as it is, even higher batch sizes than currently lead to decreases in accuracy.
2. I modified the loss function 
"""
import torch
import torchvision
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

# import torch.profiler
import gc

torch.set_num_threads(8)
torch.set_default_tensor_type("torch.cuda.FloatTensor")
train_device = torch.device("cuda")


class WeightLimit:  # Inspired by https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.clamp(-1, 1)


weightlimiter = WeightLimit()


class Binarize(object):
    def __call__(self, img):
        img[img < 0.3] = 0
        img[img >= 0.3] = 1
        return img


image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.Resize(14 + 2),
        torchvision.transforms.RandomCrop(14),
        torchvision.transforms.Resize([14, 14]),
        # torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

total_classes = 10
ds_train = torchvision.datasets.MNIST(
    "MNIST", train=True, download=True, transform=image_transform
)
ds_val = torchvision.datasets.MNIST(
    "MNIST", train=False, download=True, transform=image_transform
)
dl_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=4096,
    shuffle=True,
    generator=torch.Generator(device=train_device),
)
dl_val = torch.utils.data.DataLoader(
    ds_val,
    batch_size=2048,
    shuffle=True,
    generator=torch.Generator(device=train_device),
)

plt.title("Example image")
plt.imshow(next(enumerate(dl_train))[1][0][0][0], cmap="gray")  # white on black
plt.show()

torch.autograd.set_detect_anomaly(True)

"""def clamp_to_range(x):
	#x-=torch.min(x).item()
	#x*=1.0/torch.max(x).item()
	return x"""

range_loss_rangetens = torch.tensor(1.0).to(train_device)


# The sum of the amount by which the values go past the (-1, 1) range
def range_loss(x):
    # if torch.min(x)<-1.0 or torch.max(x)>1.0:
    # 	print(torch.min(x), torch.max(x))
    # return (torch.sum(x[x>1.0]-1)+torch.sum((-x[x<-1.0])-1))
    return torch.sum(torch.maximum(torch.abs(x), range_loss_rangetens) - 1.0)


class LeNet_Simplified(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=2, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 12, kernel_size=3, stride=1, bias=False)
        self.fc = nn.Linear(12 * 3 * 3, 10)
        self.add_loss = 0.0
        self.min_values = torch.tensor(0.0)
        self.max_values = torch.tensor(0.0)

    def forward(self, x):
        x = self.conv1(x)
        if self.training:
            with torch.no_grad():
                self.min_values = torch.tensor(0.0)
                self.max_values = torch.tensor(0.0)
                self.min_values = torch.minimum(self.min_values, torch.min(x))
                self.max_values = torch.maximum(self.max_values, torch.max(x))
            self.add_loss += range_loss(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.training:
            with torch.no_grad():
                self.min_values = torch.minimum(self.min_values, torch.min(x))
                self.max_values = torch.maximum(self.max_values, torch.max(x))
            self.add_loss += range_loss(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        """if self.training:
			with torch.no_grad():
				self.min_values=torch.minimum(self.min_values, torch.min(x))
				self.max_values=torch.maximum(self.max_values, torch.max(x))
			self.add_loss+=range_loss(x)"""
        return x


model = LeNet_Simplified()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()


def train(model, epochs, device, save_every_epochs=0):
    history = []
    model = model.to(device)
    grad_history = {}
    start_epochs = 0
    if 0:
        trainstate = torch.load("trainstate_50.pt")
        optimizer.load_state_dict(trainstate["optimizer"])
        start_epochs = trainstate["epochs"]
        model = trainstate["model"]
    for name, param in model.named_parameters():
        grad_history[name] = []
    for epoch in tqdm.tqdm(range(start_epochs, epochs)):
        train_loss = 0.0
        train_loss_add = 0.0
        train_acc = 0.0
        model.train()
        for images, labels in dl_train:
            model.add_loss = 0.0
            optimizer.zero_grad()
            img = images.to(device)
            outputs = model(img)
            loss = (
                loss_fn(outputs, labels) + model.add_loss * 0.0035
            )  # Adding the range outliers loss to force all numbers back into the (-1, 1) range
            with torch.no_grad():
                train_loss += loss.item() / labels.shape[0]
                train_loss_add += model.add_loss.item()
                train_acc += (
                    torch.sum(torch.argmax(outputs, dim=1) == labels) / labels.shape[0]
                ).item()
            loss.backward()
            optimizer.step()
        print(model.min_values, model.max_values)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_history[name].append(param.grad.detach().norm().item())
        # model.apply(weightlimiter)
        train_acc /= len(dl_train)
        train_loss /= len(dl_train)
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            model.eval()
            for images, labels in dl_val:
                outputs = model(images.to(device))
                labels = labels
                # val_loss+=loss_fn(outputs, torch.nn.functional.one_hot(labels, total_classes).type(torch.float32)).item()/labels.shape[0]
                val_loss += loss_fn(outputs, labels).item() / labels.shape[0]
                val_acc += (
                    torch.sum(torch.argmax(outputs, dim=1) == labels) / labels.shape[0]
                ).item()
        val_acc /= len(dl_val)
        val_loss /= len(dl_val)
        print("\nTRAIN LOSS ", train_loss, "VAL LOSS ", val_loss)
        print("TRAIN ACC ", train_acc, "VAL ACC ", val_acc)
        history.append([[train_acc, train_loss], [val_acc, val_loss]])
        if save_every_epochs:
            if (epoch % save_every_epochs) == 0 and epoch:
                trainstate = {
                    "optimizer": optimizer.state_dict(),
                    "model": model,
                    "epochs": epoch,
                }  # "scheduler": scheduler.state_dict(),
                torch.save(trainstate, "trainstate_%s.pt" % epoch)
    return history, grad_history


"""with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
	history, grad_history=train(model, 1, train_device, save_every_epochs=50)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
prof.export_chrome_trace("trace.json")
"""
history, grad_history = train(model, 200, train_device, save_every_epochs=50)

history = np.array(history)

torch.save(model, "mnist_trained.pt")

try:
    scr = torch.jit.script(model)
    torch.jit.save(scr, "mnist_trained_scr.pt")
except Exception as e:
    print("Torch script conversion failed", e)

example_input = torch.randn(1, 1, 14, 14, device=torch.device("cpu"))
torch.onnx.export(
    model.to(torch.device("cpu")),
    example_input.to(torch.device("cpu")),
    "mnist.onnx",
    input_names=["Image_Input"],
    output_names=["Numeral_OneHot"],
    opset_version=12,
)

plt.title("Training result")
plt.plot(history[:, 0, 0], label="Train accuracy")
plt.plot(history[:, 1, 0], label="Validation accuracy")
plt.plot(history[:, 0, 1], label="Train loss")
plt.plot(history[:, 1, 1], label="Validation loss")
plt.legend()
plt.show()

for gr_name, gr_graph in grad_history.items():
    plt.plot(gr_graph)
plt.title("Gradients")
plt.show()
