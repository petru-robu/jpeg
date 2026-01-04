# encode to a target mse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.rgb_jpeg import RGBJPEG

def encode_to_target_mse(original_image, target_mse, max_iter=20, tol=0.01):
    scale_low, scale_high = 0.1, 10.0
    best_encoded = None
    best_scale = None
    final_mse = None

    for _ in range(max_iter):
        scale = (scale_low + scale_high) / 2
        
        jpeg = RGBJPEG(original_image)
        jpeg.Q = jpeg.Q * scale
        compressed = jpeg.encode()
        comparison = jpeg.compare(compressed)
        mse = comparison["mse"]
        
        if abs(mse - target_mse) <= tol:
            best_encoded = compressed
            best_scale = scale
            final_mse = mse
            break
        elif mse < target_mse:
            scale_low = scale
        else:
            scale_high = scale

        best_encoded = compressed
        best_scale = scale
        final_mse = mse

    return best_encoded, best_scale, final_mse

if __name__ == '__main__':
    # load the image
    image_path = "./assets/png/boat.png"
    image = Image.open(image_path).convert('RGB')
    rgb_image = np.array(image)[0:512, 0:512]
    
    target_mse = 17
    print(f'Finding scale for target mse={target_mse}...')
    compressed, scale, mse = encode_to_target_mse(rgb_image, target_mse)
    
    jpeg = RGBJPEG(rgb_image)
    jpeg.Q = jpeg.Q * scale
    rec = jpeg.decode(compressed)
    comparison = jpeg.compare(compressed)
    
    print('Scale:', scale)
    print('Comparison:')
    print("Original bits:", comparison["original_bits"])
    print("Compressed bits:", comparison["compressed_bits"])
    print("Compression ratio:", comparison["compression_ratio"])
    print("Bits per pixel:", comparison["bits_per_pixel"])
    print("MSE:", comparison["mse"])
    print("PSNR:", comparison["psnr"])

    # ----- plotting -----
    rec_image = comparison["reconstructed_image"]
    w_x, w_y, zoom_x, zoom_y = 100, 100, 340, 200
    plt.figure(figsize=(12, 10))
    
    # zoom-in
    plt.subplot(1, 2, 1)
    plt.title("Original ")
    plt.imshow(rgb_image[zoom_x : zoom_x + w_x, zoom_y : zoom_y + w_y])
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed\n MSE: {comparison['mse']:.2f} obtained with scale: {scale:.2f}")
    plt.imshow(rec_image[zoom_x : zoom_x + w_x, zoom_y : zoom_y + w_y])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('./output/plots/mse_.png')
    plt.show()
    
    
    