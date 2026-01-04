# encode a rgb image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.rgb_jpeg import RGBJPEG

def rgb():
    # load the image
    image_path = "./assets/png/clouds.png"
    image = Image.open(image_path).convert('RGB')
    rgb_image = np.array(image)

    # encode the image and compare
    jpeg = RGBJPEG(rgb_image)
    compressed = jpeg.encode()
    comparison = jpeg.compare(compressed)

    print("Original bits:", comparison["original_bits"])
    print("Compressed bits:", comparison["compressed_bits"])
    print("Compression ratio:", comparison["compression_ratio"])
    print("Bits per pixel:", comparison["bits_per_pixel"])
    print("MSE:", comparison["mse"])
    print("PSNR:", comparison["psnr"])

    reconstructed = comparison["reconstructed_image"]
    w_x, w_y, zoom_x, zoom_y = 70, 70, 20, 20
    plt.figure(figsize=(12, 10))

    # ----- plotting -----
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(rgb_image)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title(f"Reconstructed\nMSE: {comparison['mse']:.2f}")
    plt.imshow(reconstructed)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Difference")
    diff = np.linalg.norm(rgb_image.astype(float) - reconstructed.astype(float), axis=2)
    plt.imshow(diff, cmap="inferno")
    plt.axis("off")

    # zoom-in
    plt.subplot(2, 3, 4)
    plt.title("Original (zoomed)")
    plt.imshow(rgb_image[zoom_x : zoom_x + w_x, zoom_y : zoom_y + w_y])
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title(f"Reconstructed\nMSE: {comparison['mse']:.2f} (zoomed)")
    plt.imshow(reconstructed[zoom_x : zoom_x + w_x, zoom_y : zoom_y + w_y])
    plt.axis("off")
    
    plt.subplot(2, 3, 6)
    plt.title("Difference")
    diff = np.linalg.norm(rgb_image.astype(float) - reconstructed.astype(float), axis=2)
    plt.imshow(diff[zoom_x : zoom_x + w_x, zoom_y : zoom_y + w_y], cmap="inferno")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('./output/plots/rgb_clouds.png')
    plt.show()

if __name__ == '__main__':
    rgb()