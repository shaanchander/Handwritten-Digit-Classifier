import torch
from torch import nn
from torchvision import transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class DigitClassifier(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Flatten(),
			nn.Linear(28 * 28, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 10),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def get_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def get_mnist_transform() -> transforms.Compose:
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(MNIST_MEAN, MNIST_STD),
	])


def get_predict_transform() -> transforms.Compose:
	return transforms.Compose([
		transforms.Grayscale(),
		transforms.Resize((28, 28)),
		transforms.ToTensor(),
		transforms.Normalize(MNIST_MEAN, MNIST_STD),
	])
