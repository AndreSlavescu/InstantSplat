import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
import matplotlib.pyplot as plt
import os

# Enable device-side assertions
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Define the rasterization settings
raster_settings = GaussianRasterizationSettings(
    image_height=512,
    image_width=512,
    tanfovx=0.5,
    tanfovy=0.5,
    bg=torch.zeros((3, 512, 512), dtype=torch.float32).cuda(),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4, dtype=torch.float32).cuda(),
    projmatrix=torch.eye(4, dtype=torch.float32).cuda(),
    sh_degree=2,
    campos=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda(),
    prefiltered=False,
    debug=False
)

# Initialize the Gaussian Rasterizer
rasterizer = GaussianRasterizer(raster_settings)

# Define the input data (minimal example data)
means3D = torch.rand((10, 3), dtype=torch.float32).cuda()
means2D = torch.rand((10, 2), dtype=torch.float32).cuda()
opacities = torch.rand((10,), dtype=torch.float32).cuda()
shs = torch.rand((10, 9), dtype=torch.float32).cuda()  # Spherical harmonics coefficients
scales = torch.rand((10, 3), dtype=torch.float32).cuda()
rotations = torch.rand((10, 4), dtype=torch.float32).cuda()  # Quaternion rotations

# Print tensor shapes and types
print("means3D:", means3D.shape, means3D.dtype)
print("means2D:", means2D.shape, means2D.dtype)
print("opacities:", opacities.shape, opacities.dtype)
print("shs:", shs.shape, shs.dtype)
print("scales:", scales.shape, scales.dtype)
print("rotations:", rotations.shape, rotations.dtype)

# Perform rasterization
color, radii = rasterizer(
    means3D=means3D,
    means2D=means2D,
    opacities=opacities,
    shs=shs,  # Only providing SHs
    scales=scales,
    rotations=rotations
)

# Save the output image
output_image = color.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
plt.imshow(output_image)
plt.savefig('output_image.png')