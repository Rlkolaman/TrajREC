import cv2
import os

# Set the parent directory containing subdirectories with images
parent_dir = "/home/pp/Desktop/datasets/trajrec_data/shanghaitech/testing/frames"  # Change to your parent directory path

# Ensure the parent directory exists
if not os.path.exists(parent_dir):
    print(f"Error: Directory {parent_dir} does not exist")
    exit()

# Get all subdirectories in the parent directory
subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
subdirs.sort()  # Sort subdirectories for consistent processing

# Process each subdirectory
for subdir in subdirs:
    subdir_path = os.path.join(parent_dir, subdir)

    # Get all .jpg files in the subdirectory
    image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".jpg")]

    # Skip if no images found
    if not image_files:
        print(f"No JPG images found in {subdir_path}")
        continue

    # Sort images alphabetically
    image_files.sort()

    # Get full paths for images
    sorted_image_paths = [os.path.join(subdir_path, f) for f in image_files]

    # Read the first image to get frame size
    first_frame = cv2.imread(sorted_image_paths[0])
    if first_frame is None:
        print(f"Error: Could not read first image in {subdir_path}")
        continue

    height, width, layers = first_frame.shape
    fps = 30

    # Set output video path (same name as subdirectory)
    output_video_path = os.path.join(parent_dir, f"{subdir}.mp4")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read and write each frame
    for img_path in sorted_image_paths:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"Warning: Could not read {img_path}")

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")