import torch
from torch import nn
from torchvision import transforms
from PIL import Image


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


class DigitCNN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(),
			nn.Linear(128, 10),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		return self.classifier(x)


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


def prepare_prediction_image(image: Image.Image) -> torch.Tensor:
	image = image.convert("L")
	image = image.resize((28, 28))
	image_tensor = transforms.ToTensor()(image)
	if image_tensor.mean() > 0.5:
		image_tensor = 1.0 - image_tensor
	return transforms.Normalize(MNIST_MEAN, MNIST_STD)(image_tensor)
