import torch
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
from PIL import Image
import numpy as np
import glob


def calculate_fid(act1, act2):
	# Calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# Calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2) ** 2.0)
	# Calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# Check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# Calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


def get_activations(files, model, batch_size=50, dims=2048, device='cuda'):
	model.eval()
	N = len(files)
	pred_arr = np.empty((N, dims))
	for i in range(0, N, batch_size):
		start = i
		end = i + batch_size
		images = [transforms.functional.to_tensor(Image.open(f).convert('RGB')) for f in files[start:end]]
		images = torch.stack(images).to(device)
		with torch.no_grad():
			pred = model(images)[0]
		pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)
	return pred_arr


# Paths to directories holding real and generated images
path_real = 'path_to_real_images'
path_fake = 'path_to_generated_images'

# Model and transformations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = inception_v3(pretrained=True).to(device)
model.eval()

# Get activations
real_files = glob.glob(path_real + '/*.jpg')
fake_files = glob.glob(path_fake + '/*.jpg')
act_real = get_activations(real_files, model, device=device)
act_fake = get_activations(fake_files, model, device=device)

# Calculate FID
fid_value = calculate_fid(act_real, act_fake)
print('FID:', fid_value)
