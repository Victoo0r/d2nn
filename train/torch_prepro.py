import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
# mnist
from torchvision import datasets, transforms

# Preprocessing runs on CPU (the full-scale u0 arrays do not fit into an 8 GB GPU)
device = torch.device("cpu")

# Fixed output directory: train/preprocessed (relative to this script, independent of CWD)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed MNIST directory: train/data (shared, avoids re-downloading depending on CWD)
MNIST_DIR = os.path.join(HERE, "data")

# load the data here
# load the mnist data
mnist_data = datasets.MNIST(MNIST_DIR, train=True, download=True, transform=transforms.ToTensor())
mnist_label = mnist_data.targets
# load the mnist test data
mnist_test_data = datasets.MNIST(MNIST_DIR, train=False, download=True, transform=transforms.ToTensor())
test_data = mnist_test_data.data.to(device)
test_label = mnist_test_data.targets.to(device)
# take the first 5000 samples as validation
validation_data = mnist_data.data[:5000].to(device)
validation_label = mnist_label[:5000].to(device)
# take the rest as training data
training_data = mnist_data.data[5000:].to(device)
training_label = mnist_label[5000:].to(device)

# normalize uint8 [0,255] -> float [0,1] (binarization happens later in place_digit)
training_data = training_data.float() / 255.0
validation_data = validation_data.float() / 255.0
test_data = test_data.float() / 255.0

print(training_data.shape)
print(test_data.shape)

N = 200
lmbda = 0.75e-3
L = 0.08
w = 0.02
z = 0.03

# number of physical pixels occupied by one digit (digit width = 2w)
pixel_pitch = L / N
digit_px = round(2 * w / pixel_pitch)

# change this to 1 if want to use all the data
ratio = 1
training_data = training_data[:int(ratio * training_data.shape[0])]
test_data = test_data[:int(ratio * test_data.shape[0])]
validation_data = validation_data[:int(ratio * validation_data.shape[0])]
training_label = training_label[:int(ratio * training_label.shape[0])]
test_label = test_label[:int(ratio * test_label.shape[0])]
validation_label = validation_label[:int(ratio * validation_label.shape[0])]

# one-hot labels (dim = 10), vectorized
train_label_mod = F.one_hot(training_label, 10).float()
validation_label_mod = F.one_hot(validation_label, 10).float()
test_label_mod = F.one_hot(test_label, 10).float()

print("Finish loading label")
train_data_num = training_data.shape[0]
test_data_num = test_data.shape[0]
validation_data_num = validation_data.shape[0]

u0_train = torch.zeros((train_data_num, N, N), dtype=torch.float32, device=device)
u0_test = torch.zeros((test_data_num, N, N), dtype=torch.float32, device=device)
u0_validation = torch.zeros((validation_data_num, N, N), dtype=torch.float32, device=device)

def place_digit(img28):
    # bicubic up-sample the 28x28 digit to its physical size, then center it
    up = F.interpolate(img28[None, None, ...], size=(digit_px, digit_px),
                       mode='bicubic', align_corners=False)[0, 0]
    up = up.clamp(0.0, 1.0)
    # hard binarization after smooth up-sampling (threshold on [0,1])
    up = (up >= 0.2).float()
    canvas = torch.zeros(N, N, dtype=torch.float32, device=device)
    c0 = (N - digit_px) // 2
    canvas[c0:c0 + digit_px, c0:c0 + digit_px] = up
    return canvas

print("Begin processing data")
for t in tqdm(range(train_data_num)):
    u0_train[t, :, :] = place_digit(training_data[t])

for t in tqdm(range(test_data_num)):
    u0_test[t, :, :] = place_digit(test_data[t])

for t in tqdm(range(validation_data_num)):
    u0_validation[t, :, :] = place_digit(validation_data[t])

# randomly select a sample to visualize

index = np.random.randint(0, train_data_num)
plt.imshow(u0_train[index].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
plt.savefig(os.path.join(OUT_DIR, "u0_train.png"))
plt.close()
print(train_label_mod[index])

print('Finish preprocessing, now saving to files')
np.save(os.path.join(OUT_DIR, 'u0_train_all.npy'), u0_train.cpu().detach().numpy())
np.save(os.path.join(OUT_DIR, 'u0_test_all.npy'), u0_test.cpu().detach().numpy())
np.save(os.path.join(OUT_DIR, 'u0_validation_all.npy'), u0_validation.cpu().detach().numpy())
np.save(os.path.join(OUT_DIR, 'train_label_all.npy'), train_label_mod.cpu().detach().numpy())
np.save(os.path.join(OUT_DIR, 'test_label_all.npy'), test_label_mod.cpu().detach().numpy())
np.save(os.path.join(OUT_DIR, 'validation_label_all.npy'), validation_label_mod.cpu().detach().numpy())
