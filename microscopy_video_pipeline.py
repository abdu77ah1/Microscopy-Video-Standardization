import cv2
import numpy as np
from PIL import Image
import os

def process_tif_video():
    # --- HARD-CODED PATHS ---
    tif_file_path = r"C:\Master Thesis\Experimentation\6 Months\Myoblast__8_MMStack_Pos0-Puit-Haut-Gauche.ome_processed.tiff"
    converted_folder_path = r"C:\Master Thesis\Experimentation\6 Months\converted"
    processed_folder_path = r"C:\Master Thesis\Experimentation\6 Months\processed"

    # Create folders if they don't exist
    os.makedirs(converted_folder_path, exist_ok=True)
    os.makedirs(processed_folder_path, exist_ok=True)

    fps = 30  # Output video FPS
    crop_px = 10  # Pixels to crop after stabilization

    # --- PART 1: Convert TIFF to 8-bit frames ---
    tif = Image.open(tif_file_path)
    width, height = tif.size

    converted_frames = []
    frame_idx = 0
    try:
        while True:
            frame = np.array(tif, dtype=np.float32)
            frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)

            # Ensure single-channel grayscale
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            converted_frames.append(frame_bgr)
            frame_idx += 1
            tif.seek(frame_idx)
    except EOFError:
        pass

    # Save converted video (pre-stabilization)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    converted_video_path = os.path.join(
        converted_folder_path,
        f"{os.path.splitext(os.path.basename(tif_file_path))[0]}_converted.mp4"
    )
    out = cv2.VideoWriter(converted_video_path, fourcc, fps, (width, height))
    for frame in converted_frames:
        out.write(frame)
    out.release()

    # --- PART 2: Stabilize video using SIFT ---
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    stabilized_frames = []

    prev_gray = cv2.cvtColor(converted_frames[0], cv2.COLOR_BGR2GRAY)
    # Crop first frame
    first_frame_cropped = converted_frames[0][crop_px:-crop_px, crop_px:-crop_px]
    stabilized_frames.append(first_frame_cropped)
    expected_height, expected_width = first_frame_cropped.shape[:2]

    for i in range(1, len(converted_frames)):
        curr_frame = converted_frames[i]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(prev_gray, None)
        kp2, des2 = sift.detectAndCompute(curr_gray, None)

        if des1 is not None and des2 is not None:
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                m, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if m is not None:
                    curr_frame = cv2.warpAffine(curr_frame, m, (width, height))

        # Crop edges and resize to expected size
        curr_frame = curr_frame[crop_px:-crop_px, crop_px:-crop_px]
        curr_frame = cv2.resize(curr_frame, (expected_width, expected_height))
        stabilized_frames.append(curr_frame)
        prev_gray = curr_gray

    # --- PART 3: Global brightness normalization ---
    brightness_sum = 0
    for frame in stabilized_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_sum += np.mean(gray)

    global_brightness = brightness_sum / len(stabilized_frames)

    final_frames = []
    for frame in stabilized_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        brightness_scale = global_brightness / current_brightness if current_brightness > 0 else 1.0
        frame_adjusted = cv2.convertScaleAbs(frame, alpha=brightness_scale, beta=0)

        # Resize again to ensure consistent dimensions
        frame_adjusted = cv2.resize(frame_adjusted, (expected_width, expected_height))
        final_frames.append(frame_adjusted)

    # --- PART 4: Save final processed video ---
    final_video_path = os.path.join(
        processed_folder_path,
        f"{os.path.splitext(os.path.basename(tif_file_path))[0]}_processed.mp4"
    )
    out = cv2.VideoWriter(final_video_path, fourcc, fps, (expected_width, expected_height))
    for frame in final_frames:
        out.write(frame)
    out.release()

    print(f"Processed video saved to {final_video_path}")
    return global_brightness

# --- CALL FUNCTION ---
process_tif_video()
