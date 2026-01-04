import numpy as np
from scipy.fft import dctn, idctn
from src.huffman import HuffmanCoder
import pickle

# Quantization matrix
Q_jpeg = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
]

class JPEG:
    def __init__(self, original, Q_matrix=Q_jpeg):
        self.original = original.astype(np.float32)
        self.H, self.W = self.original.shape
        self.Q = np.array(Q_matrix)        
        self.block_size = 8                        

    def preprocess(self):
        pad_H = (self.block_size - self.H % self.block_size) % self.block_size
        pad_W = (self.block_size - self.W % self.block_size) % self.block_size
        img = np.pad(self.original, ((0, pad_H), (0, pad_W)), mode="constant")
        return img - 128  # shift [0,255] range to [-128, 127]

    def encode(self):
        img = self.preprocess()
        H, W = img.shape
        symbols = []
        
        # Process each 8x8 block
        for i in range(0, H, self.block_size):
            for j in range(0, W, self.block_size):
                block = img[i:i+8, j:j+8]

                # 1) Apply DCT to the block
                dct = dctn(block, type=2, norm="ortho")

                # 2) Quantize DCT coefficients
                q = np.round(dct / self.Q).astype(int)

                # 3) Flatten 
                symbols.extend(q.flatten())

        # 4) Build Huffman tree and encode
        huff = HuffmanCoder()
        huff.build(symbols)
        bitstream = huff.encode(symbols)

        return {
            "bitstream": bitstream,
            "huff_instance": huff,
            "shape": (self.H, self.W),
            "Q": self.Q,
            "block_size": self.block_size,
        }

    def decode(self, compressed):
        H, W = compressed["shape"]
        Q = compressed["Q"]
        block_size = compressed["block_size"]

        bitstream = compressed["bitstream"]
        huff = compressed["huff_instance"]
        symbols = huff.decode(bitstream)

        blocks_per_row = (W + block_size - 1) // block_size
        blocks_per_col = (H + block_size - 1) // block_size

        img_rec = np.zeros((blocks_per_col * block_size,
                            blocks_per_row * block_size))
        idx = 0
        for i in range(blocks_per_col):
            for j in range(blocks_per_row):
                block_flat = symbols[idx:idx + block_size * block_size]
                idx += block_size * block_size

                block = np.array(block_flat).reshape((block_size, block_size))
                block = block * Q
                block_rec = idctn(block, type=2, norm="ortho")

                img_rec[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ] = block_rec

        img_rec = img_rec[:H, :W] + 128
        img_rec = np.clip(np.round(img_rec), 0, 255).astype(np.uint8)
        return img_rec


    def compare(self, compressed):
        orig_bits = self.H * self.W * 8                  
        compressed_bits = len(compressed["bitstream"])
        compression_ratio = orig_bits / compressed_bits
        bits_per_pixel = compressed_bits / (self.H * self.W)
        
        rec = self.decode(compressed)
        mse = np.mean((self.original - rec) ** 2)
        psnr = 10 * np.log10(255**2 / mse)

        return {
            "original_bits": orig_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "bits_per_pixel": bits_per_pixel,
            "mse": mse,
            "psnr": psnr,
            "reconstructed_image": rec,
        }
        
    def encode_to_file(self, filepath):
        compressed = self.encode()
        with open(filepath, "wb") as f:
            pickle.dump(compressed, f)
            
    def decode_from_file(self, filepath):
        with open(filepath, "rb") as f:
            compressed = pickle.load(f)

        dummy = JPEG(
            original=np.zeros(compressed["shape"], dtype=np.uint8),
            Q_matrix=compressed["Q"]
        )
        
        return dummy.decode(compressed)