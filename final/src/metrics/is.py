import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np


def inception_score(img_path1, img_path2):
	transform = transforms.Compose([
		transforms.Resize((299, 299)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# Load images and apply transformations
	img1 = transform(Image.open(img_path1).convert('RGB')).unsqueeze(0)
	img2 = transform(Image.open(img_path2).convert('RGB')).unsqueeze(0)
	imgs = torch.cat([img1, img2], dim=0)

	# Load pre-trained Inception v3 model
	inception_model = models.inception_v3(pretrained=True, transform_input=False)
	inception_model.eval()

	# Predictions
	with torch.no_grad():
		preds = inception_model(imgs)
	preds = torch.nn.functional.softmax(preds, dim=1).numpy()

	# Calculate the Inception Score
	kl_div = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
	kl_div = np.mean(np.sum(kl_div, 1))
	is_score = np.exp(kl_div)

	return is_score


# Replace 'image1.jpg' and 'image2.jpg' with your image file paths
score = inception_score('image1.jpg', 'image2.jpg')
print(f"Inception Score: {score}")
