import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from functions import get_device, get_mnist_transform


EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SAVE_PATH = "model-cnn.pt"


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


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
	model.train()
	total_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		logits = model(images)
		loss = loss_fn(logits, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * labels.size(0)
		predictions = logits.argmax(dim=1)
		correct += (predictions == labels).sum().item()
		total += labels.size(0)

	return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		logits = model(images)
		loss = loss_fn(logits, labels)

		total_loss += loss.item() * labels.size(0)
		predictions = logits.argmax(dim=1)
		correct += (predictions == labels).sum().item()
		total += labels.size(0)

	return total_loss / total, correct / total


def build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
	transform = get_mnist_transform()

	train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
	test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader


def main() -> None:
	torch.manual_seed(42)

	device = get_device()
	print(f"Using device: {device}")

	train_loader, test_loader = build_loaders(batch_size=BATCH_SIZE)

	model = DigitCNN().to(device)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	for epoch in range(1, EPOCHS + 1):
		train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
		test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

		print(
			f"Epoch {epoch}/{EPOCHS} | "
			f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
			f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}"
		)

	torch.save(model.state_dict(), SAVE_PATH)
	print(f"Saved model weights to {SAVE_PATH}")


if __name__ == "__main__":
	main()
