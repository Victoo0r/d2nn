import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------Optical parameters---------------------
L = 0.08
lmbda = 0.75e-3
M = 200
w = 0.02
z = 0.03
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)


# ---------------------spliting the output area--------------------
square_size = 12  # 4.8 mm detector width (12 pixels x 0.4 mm)
canvas_size = M

# define the places for each digit
positions = np.array([[0, 1], [0, 4], [0, 7], [3, 0.65], [3, 2.65+0.5], [3, 4.65+1], [3, 6.65+1.5], [6, 1], [6, 4], [6, 7]])

# Calculate the offset to center the squares in the canvas
offset = (canvas_size - 8 * square_size) / 2

start_col = np.zeros(positions.shape[0], dtype=int)
start_row = np.zeros(positions.shape[0], dtype=int)

for i in range(10):
    row, col = positions[i]
    start_row[i] = round(offset + row * square_size)
    start_col[i] = round(offset + col * square_size)

def count_pixel(img, start_x, start_y, width, height):
    # Slicing the batch of images and summing over the desired region for each sample
    return torch.sum(img[:, start_x:start_x+width, start_y:start_y+height], dim=(1, 2))

class propagation_layer(nn.Module):
    def __init__(self, L, lmbda, z):
        super(propagation_layer, self).__init__()
        self.L = L
        self.lmbda = lmbda
        self.z = z

        # Build the fixed transfer function once (it does not depend on the input)
        dx = L / M
        k = 2 * np.pi / lmbda
        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, M,
                            device=device, dtype=torch.float64)
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')
        KX = 2 * np.pi * FX
        KY = 2 * np.pi * FY
        # Exact Rayleigh-Sommerfeld angular spectrum (no Fresnel/paraxial expansion):
        # kz = sqrt(k^2 - kx^2 - ky^2); real for propagating, imaginary for evanescent
        kz = torch.sqrt((k**2 - KX**2 - KY**2) + 0j)
        self.H = torch.fft.fftshift(torch.exp(j * kz * z), dim=(-2, -1))

    def forward(self, u1):
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2, -1)))
        U2 = self.H * U1
        return torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2, -1))

class modulation_layer(nn.Module):
    def __init__(self, M, N):
        super(modulation_layer, self).__init__()
        # Unconstrained trainable parameter; sigmoid maps it to a phase bounded to [0, pi]
        self.theta = nn.Parameter(torch.zeros((M, N)))

    def forward(self, input_tensor):
        # phase = pi * sigmoid(theta) in [0, pi], phase-only modulation
        phase = np.pi * torch.sigmoid(self.theta)
        return input_tensor * torch.exp(j * phase)

class imaging_layer(nn.Module):
    def __init__(self):
        super(imaging_layer, self).__init__()

    def forward(self, u):
        intensity = torch.abs(u)**2
        # Gather the ten detector signals (batch x 10), then vectorized normalization
        values = torch.stack([count_pixel(intensity, start_row[i], start_col[i],
                                          square_size, square_size)
                              for i in range(10)], dim=1)
        norms = torch.sqrt((values**2).sum(dim=1, keepdim=True) + 1e-9)
        return values / norms

class OpticalNetwork(nn.Module):
    def __init__(self, M, L, lmbda, z, layersCount):
        super(OpticalNetwork, self).__init__()

        # Propagation and Modulation layers interleaved
        layers = []
        for _ in range(layersCount):
            layers.append(propagation_layer(L, lmbda, z))
            layers.append(modulation_layer(M, M))

        self.optical_layers = nn.Sequential(*layers)
        self.imaging = imaging_layer()

    def forward(self, x):
        x = self.optical_layers(x)
        return self.imaging(x)

    # Same inference, but also returns the detector-plane intensity for visualization
    def forward_with_intensity(self, x):
        u = self.optical_layers(x)
        intensity = torch.abs(u)**2
        return self.imaging(u), intensity[0].cpu().detach().numpy()
