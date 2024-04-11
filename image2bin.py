from PIL import Image
import numpy as np

# Step 1: Read the image using PIL
image = Image.open("/home/bozhan/repo/mscompression_official/test/CLDLOW_1_1800_3600.png")

# Convert the image to a NumPy array with dtype as uint8
image_np = np.array(image, dtype=np.uint8)

# Check if the image has an alpha channel and remove it if present
if image_np.shape[2] == 4:
    image_np = image_np[:, :, :3]

# Step 2: Convert the format from (x, y, channel) to (channel, x, y)
image_np_transposed = np.transpose(image_np, (2, 0, 1))

# Ensure the data type is uint8 for the transposed array as well (should already be uint8 if the original was)
image_np_transposed = image_np_transposed.astype(np.uint8)

# Step 3: Write the image as a binary file specifying the dtype as uint8
binary_file_path = "/home/bozhan/repo/mscompression_official/test/CLDLOW_1_1800_3600.bin"
image_np_transposed.tofile(binary_file_path)

print(f"Image written as binary file with dtype uint8: {binary_file_path}")
