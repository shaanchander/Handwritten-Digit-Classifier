import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_IMAGE_SIZE = 28
MNIST_DIGIT_TARGET_SIZE = 22
FOREGROUND_THRESHOLD = 40	# just in case there's aliasing or anything


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


def _mnist_resample_mode() -> int:
	return getattr(Image, "Resampling", Image).LANCZOS


def _extract_centered_digit(image: Image.Image) -> Image.Image:
	grayscale = image.convert("L")

	# canvas uses dark ink on white background, need to invert to match MNIST
	inverted = ImageOps.invert(grayscale)
	mask = inverted.point(lambda value: 255 if value > FOREGROUND_THRESHOLD else 0)
	bbox = mask.getbbox()

	if bbox is None:
		return Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), color=0)

	digit = inverted.crop(bbox)
	width, height = digit.size
	longest_side = max(width, height)
	scale = MNIST_DIGIT_TARGET_SIZE / float(longest_side)
	resized_width = max(1, int(round(width * scale)))
	resized_height = max(1, int(round(height * scale)))
	resized_digit = digit.resize((resized_width, resized_height), _mnist_resample_mode())

	canvas = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), color=0)
	left = (MNIST_IMAGE_SIZE - resized_width) // 2
	top = (MNIST_IMAGE_SIZE - resized_height) // 2
	canvas.paste(resized_digit, (left, top))
	return canvas


def prepare_prediction_image(image: Image.Image) -> torch.Tensor:
	image = _extract_centered_digit(image)
	image_tensor = transforms.ToTensor()(image)
	return transforms.Normalize(MNIST_MEAN, MNIST_STD)(image_tensor)
