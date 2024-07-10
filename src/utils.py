import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import os


def mkdirs(dir_path: str, remove_flag: bool = False):
    """
    Creates a directory at the specified `dir_path` if it does not already exist.
    
    Args:
        dir_path (str): The path of the directory to create.
        remove_flag (bool, optional): If True, removes the directory at `dir_path` if it already exists. Defaults to False.
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If the parent directory of `dir_path` does not exist.
        PermissionError: If the user does not have permission to create the directory.
    
    Prints:
        - "Directory already exists" if the directory already exists.
        - "Directory removed" if the directory already exists and `remove_flag` is True.
        - "Directory created successfully" after the directory is created.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, mode=0o777)
        print("Directory created successfully")
    else:
        print("Directory already exists")
        if remove_flag:
            shutil.rmtree(dir_path)
            print("Directory removed") 
            os.makedirs(dir_path, mode=0o777)
            print("Directory created successfully")


def traverse_folder(folder_path: str, startswith: tuple, endswith: tuple = ('.mp4', '.avi', '.mov', '.mkv')):
    """
    Traverse a folder and return a list of all video files (with extensions .mp4, .avi, .mov, .mkv) found in the folder (include its subfolder).

    Parameters:
        folder_path (str): The path of the folder to traverse.

    Returns:
        list: A list of strings representing the paths of all video files found in the folder.
    """
    videos_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(endswith) and file.startswith(startswith):
                video_path = os.path.join(root, file)
                # print(video_path)
                videos_list.append(video_path)
                # video2frame(video_path, output_dir)
    return videos_list


def video2frame(video_path: str, output_dir: str, save_freq: int = 1):
    """
    Convert a video file to individual frames and save them as images in the specified output directory.

    Parameters:
        video_path (str): The path to the video file.
        output_dir (str): The directory where the frames will be saved.
        save_freq (int, optional): The frequency at which frames will be saved. Defaults to 1.

    Returns:
        None

    This function uses OpenCV to read the video file and extract individual frames. It calculates the video's properties
    such as frames per second (fps), width, height, and total number of frames. It then processes each frame and saves
    it as an image in the specified output directory. The frames are saved with a filename format of "{fileName}_{frame_count:06d}.jpg".
    The progress of the frame extraction is displayed using a progress bar.

    Example usage:
        video_path = '/path/to/video.mp4'
        output_dir = '/path/to/output/directory'
        video2frame(video_path, output_dir)
    """
    video = cv2.VideoCapture(video_path)
    fileName = video_path.split('/')[-1].split('.')[0]

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Fps, width, height and total_frames of the video: {fps}, {width}, {height}, and {total_frames}')

    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds // 60
    duration_seconds %= 60
    print(f'Time-length of the video: {int(duration_minutes)} minutes and {int(duration_seconds)} seconds')

    pbar = tqdm(total=int(total_frames))
    # Process each frame
    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame as an image
        if frame_count % save_freq == 0:
            frame_path = os.path.join(output_dir, f'{fileName}_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            # Output information

        # print(f'Frame {frame_count}/{total_frames}: size={width}x{height}, fps={fps:.2f}')

        # Increment frame count
        frame_count += 1
        pbar.update(1)
    # Release the video file
    video.release()
    pbar.close()

