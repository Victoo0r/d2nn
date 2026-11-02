import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from helper import device, M, L, lmbda, z, square_size, start_row, start_col, OpticalNetwork

# Fixed paths relative to this script (independent of the working directory)
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "train", "weights_large_uniform.pt")
TEST_DATA = os.path.join(HERE, "..", "train", "preprocessed", "u0_test_all.npy")
TEST_LABELS = os.path.join(HERE, "..", "train", "preprocessed", "test_label_all.npy")

print(f"Using {device} device")

# Build the network and load the newly trained weights
model = OpticalNetwork(M, L, lmbda, z, layersCount=5).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=torch.device(device)))
model.eval()

test_data = np.load(TEST_DATA)
test_labels = np.load(TEST_LABELS)
# Move to device and cast to complex128 (the network uses double-precision complex fields)
test_data = torch.from_numpy(test_data).double().to(device).to(torch.complex128)

# Pick a sample from the 10,000-image test set
index = int(input("please input an integer index in range(10000): "))
u0 = test_data[index]
true_label = int(np.argmax(test_labels[index]))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(u0.cpu().detach().numpy().real, cmap='gray', vmin=0, vmax=1)
plt.title("input optical field")

with torch.no_grad():
    output, inten = model.forward_with_intensity(u0.unsqueeze(0))

print("normalized detector signals:", output.data.cpu().numpy().round(4))
predicted = int(torch.argmax(output, dim=1).item())

plt.subplot(1, 2, 2)
ax = plt.gca()
for i in range(10):
    rect = Rectangle((start_col[i], start_row[i]), square_size, square_size,
                     fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(rect)
plt.imshow(inten)
plt.title("imaging intensity")
plt.savefig(os.path.join(HERE, "combined_figure.png"))
plt.close()

print(f"true label: {true_label}, predicted: {predicted}")
