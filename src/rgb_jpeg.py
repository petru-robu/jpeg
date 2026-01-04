import numpy as np
import pickle
from scipy.fft import dctn, idctn
from src.huffman import HuffmanCoder

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

class RGBJPEG:
    def __init__(self, original, Q_matrix=Q_jpeg):
        self.original = original.astype(np.float32)
        self.H, self.W, _ = self.original.shape
        self.Q = np.array(Q_matrix)
        self.block_size = 8

    def rgb_to_ycrcb(self, img):
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        Y  = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 128
        Cb = (B - Y) * 0.564 + 128
        return np.stack([Y, Cr, Cb], axis=-1)

    def ycrcb_to_rgb(self, img):
        Y, Cr, Cb = img[..., 0], img[..., 1], img[..., 2]
        R = Y + 1.403 * (Cr - 128)
        B = Y + 1.773 * (Cb - 128)
        G = (Y - 0.299 * R - 0.114 * B) / 0.587
        return np.clip(np.stack([R, G, B], axis=-1), 0, 255)

    def preprocess_channel(self, channel):
        # pad each channel and shift
        pad_H = (self.block_size - self.H % self.block_size) % self.block_size
        pad_W = (self.block_size - self.W % self.block_size) % self.block_size
        padded = np.pad(channel, ((0, pad_H), (0, pad_W)), mode="constant")
        return padded - 128
    
    def encode_channel(self, channel):
        # encode a channel
        padded = self.preprocess_channel(channel)
        H_pad, W_pad = padded.shape
        symbols = []

        for i in range(0, H_pad, self.block_size):
            for j in range(0, W_pad, self.block_size):
                block = padded[i:i+self.block_size, j:j+self.block_size]
                dct_block = dctn(block, type=2, norm="ortho")
                q_block = np.round(np.array(dct_block) / self.Q).astype(int)
                symbols.extend(q_block.flatten())

        huff = HuffmanCoder()
        huff.build(symbols)
        bitstream = huff.encode(symbols)

        return {
            "bitstream": bitstream,
            "huffman_table": huff.codes,
            "huff_instance": huff,
            "shape": (self.H, self.W)
        }

    def encode(self):
        # transform to ycrcb and encode every channel
        ycrcb = self.rgb_to_ycrcb(self.original)
        encoded_y  = self.encode_channel(ycrcb[..., 0])
        encoded_cr = self.encode_channel(ycrcb[..., 1])
        encoded_cb = self.encode_channel(ycrcb[..., 2])
        return {"Y": encoded_y, "Cr": encoded_cr, "Cb": encoded_cb}

    def decode_channel(self, compressed):
        # decode a channel
        bitstream = compressed["bitstream"]
        H, W = compressed["shape"]
        huff = compressed["huff_instance"]
        symbols = huff.decode(bitstream)

        block_size = self.block_size
        blocks_per_row = (W + block_size - 1) // block_size
        blocks_per_col = (H + block_size - 1) // block_size

        img_rec = np.zeros((blocks_per_col * block_size, blocks_per_row * block_size))
        idx = 0
        for i in range(blocks_per_col):
            for j in range(blocks_per_row):
                block_flat = symbols[idx:idx + block_size*block_size]
                block = np.array(block_flat).reshape((block_size, block_size))
                idx += block_size * block_size
                block = block * self.Q
                block_rec = idctn(block, type=2, norm="ortho")
                img_rec[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block_rec

        return np.clip(np.round(img_rec[:H, :W] + 128), 0, 255)
    
    def decode(self, compressed):
        # decode each channel
        y = self.decode_channel(compressed["Y"])
        cr = self.decode_channel(compressed["Cr"])
        cb = self.decode_channel(compressed["Cb"])
        ycrcb = np.stack([y, cr, cb], axis=-1)
        return self.ycrcb_to_rgb(ycrcb).astype(np.uint8)

    def compare(self, compressed):
        # metrics
        rec = self.decode(compressed)
        
        orig_bits = self.H * self.W * 24
        compressed_bits = sum(len(c["bitstream"]) for c in compressed.values())
        compression_ratio = orig_bits / compressed_bits
        
        bits_per_pixel = compressed_bits / (self.H * self.W)
        mse = np.mean((self.original - rec) ** 2)
        psnr = 10 * np.log10(255**2 / mse)
        
        return {
            "original_bits": orig_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "bits_per_pixel": bits_per_pixel,
            "mse": mse,
            "psnr": psnr,
            "reconstructed_image": rec
        }
        
    def encode_to_file(self, filepath):
        compressed = self.encode()
        payload = {
            "compressed": compressed,
            "shape": (self.H, self.W),
            "Q": self.Q,
            "block_size": self.block_size
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def decode_from_file(self, filepath):
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        compressed = payload["compressed"]

        dummy = RGBJPEG(
            original=np.zeros((payload["shape"][0],
                               payload["shape"][1], 3), dtype=np.uint8),
            Q_matrix=payload["Q"]
        )
        dummy.block_size = payload["block_size"]
        return dummy.decode(compressed)
