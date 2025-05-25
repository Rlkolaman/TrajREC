import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_frames(video_path, output_dir, frame_prefix=''):
    """
    Extract frames from a video and save them as JPEG files in the output directory using OpenCV.

    Args:
        video_path (str): Path to the input video file (.avi).
        output_dir (str): Directory to save the extracted frames.
        frame_prefix (str): Prefix for frame filenames (e.g., '001.jpg').
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        # Get total number of frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Iterate through frames
        frame_count = 0
        with tqdm(total=total_frames, desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Generate frame filename (e.g., 001.jpg)
                frame_filename = f"{frame_prefix}{frame_count + 1:03d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                # Save frame as JPEG
                cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                frame_count += 1
                pbar.update(1)

        logger.info(f"Extracted {frame_count} frames from {video_path} to {output_dir}")
        return frame_count

    except Exception as e:
        logger.error(f"Failed to process {video_path}: {str(e)}")
        return 0
    finally:
        if 'cap' in locals():
            cap.release()


def generate_training_frames(dataset_root, video_dir='training/videos', output_dir='training/frames'):
    """
    Scan the training video directory and generate frames for each video.

    Args:
        dataset_root (str): Root directory of the ShanghaiTech dataset.
        video_dir (str): Relative path to the training videos directory.
        output_dir (str): Relative path to the output frames directory.
    """
    # Resolve paths
    video_dir_path = Path(dataset_root) / video_dir
    output_dir_path = Path(dataset_root) / output_dir

    if not video_dir_path.exists():
        logger.error(f"Video directory {video_dir_path} does not exist.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir_path, exist_ok=True)

    # Scan for .avi files
    video_files = sorted([f for f in video_dir_path.glob('*.avi')])
    if not video_files:
        logger.warning(f"No .avi files found in {video_dir_path}")
        return

    logger.info(f"Found {len(video_files)} video files in {video_dir_path}")

    # Process each video
    for video_path in video_files:
        # Extract video identifier (e.g., '01_0014' from '01_0014.avi')
        video_id = video_path.stem

        # Create output subdirectory for this video
        video_output_dir = output_dir_path / video_id
        os.makedirs(video_output_dir, exist_ok=True)

        # Extract frames
        frame_count = extract_frames(str(video_path), str(video_output_dir))
        if frame_count == 0:
            logger.warning(f"No frames extracted for {video_path}")
        else:
            logger.info(f"Completed processing {video_id} with {frame_count} frames")


def main():
    # Specify the root directory of the ShanghaiTech dataset
    dataset_root = '/home/pp/Desktop/datasets/trajrec_data/shanghaitech'  # Update this path

    logger.info("Starting frame extraction for training videos")
    generate_training_frames(dataset_root)
    logger.info("Frame extraction completed")


if __name__ == '__main__':
    main()