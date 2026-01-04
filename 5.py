# saving encoded file / opening and decoding
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.jpeg import JPEG

if __name__ == "__main__":
    # load the image
    image_path = "./assets/png/boat.png"
    image = Image.open(image_path)
    gray_image = np.array(image.convert("L"))
    
    save_path = "./output/myjpeg/boat.myjpeg"
    
    # encode and save
    jpeg = JPEG(gray_image)
    jpeg.Q *= 2
    jpeg.encode_to_file(save_path)
    print(f"Encoding complete. File saved as {save_path}")

    # decode from file
    reconstructed = jpeg.decode_from_file(save_path)

    print("Decoding complete.")
    # plotting
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    

