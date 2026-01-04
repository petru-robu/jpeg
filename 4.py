# compress video
import cv2 
import matplotlib.pyplot as plt
from src.rgb_jpeg import RGBJPEG

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

def build_video(frames, output_path, fps=30):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_rgb in frames:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return output_path

if __name__ == '__main__':
    video_path = "./assets/mp4/video.mp4"
    frames = extract_frames(video_path)

    print("Encoding every frame...")
    encoded_frames = []
    for frame in frames:
        jpeg = RGBJPEG(frame)
        jpeg.Q *= 2
        encoded = jpeg.encode()
        comparison = jpeg.compare(encoded)
        reconstructed = comparison["reconstructed_image"]
        
        encoded_frames.append(reconstructed)
    
    print("Building encoded video...")
    
    output_path = "./output/output.mp4"
    build_video(encoded_frames, output_path)
    
    print(f"Saved encoded video as {output_path}")
    
    frame_no = 15
    raw_frame = frames[frame_no]
    compressed_frame = encoded_frames[frame_no]
    
    # ----- plotting -----
    plt.figure(figsize=(12, 10))
    
    # zoom-in
    plt.subplot(1, 2, 1)
    plt.title(f"Original frame no. {frame_no}")
    plt.imshow(raw_frame)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed frame no. {frame_no}")
    plt.imshow(compressed_frame)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('./output/plots/video_comparison.png')
    plt.show()