from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from src.utils import mkdirs
from src.ViBe import ViBe
import numpy as np
import cupy as cp
import tqdm
import time
import cv2

# from line_profiler import profile

WIDTH = 720
HEIGHT = 576


def getVideoInfo(video_caption):
    fps = video_caption.get(cv2.CAP_PROP_FPS)  # 5
    width = video_caption.get(cv2.CAP_PROP_FRAME_WIDTH)  # 3
    height = video_caption.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 4
    frame_count = video_caption.get(cv2.CAP_PROP_FRAME_COUNT)  # 7

    return fps, int(width), int(height), int(frame_count)


def main(input_path, output_path):
    video_cap = cv2.VideoCapture(input_path)
    fps, width, height, frame_count = getVideoInfo(video_cap)
    print("fps: {}, width: {}, height: {}, frames' num: {}".format(fps, width, height,frame_count))

    re_width = WIDTH
    re_height = HEIGHT

    # background model -Vibe
    vibe = ViBe(re_width, re_height)

    results_list = []
    
    i = 0
    all_count = 0
    time_interval = 1
    while True:
        ret = video_cap.grab()
        if not ret:
            break

        all_count += 1
        if all_count % time_interval == 0:
            ret, frame = video_cap.retrieve()
            if frame is None:
                break
            
            frame = cv2.resize(frame, (re_width, re_height))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if i == 0:
                vibe.processFirstFrame(gray_frame)
                cp_binary, white_count = vibe.getBGmodel()
            else:
                vibe.updateBGmodel(gray_frame)
                cp_binary, white_count = vibe.getBGmodel()
            
            np_binary = cp.asnumpy(cp_binary)
            np_binary = np.tile(np_binary[..., np.newaxis], (1, 1, 3))
            img_merge = np.concatenate((frame, np_binary), axis=1)
            results_list.append(img_merge)
            i += 1

    # initialize video parameters
    frame_size = (re_width * 2, re_height) 
    fps = int(fps)

    # create the video writer
    video_writer = FFMPEG_VideoWriter(output_path, fps=fps, size=frame_size)
    for i, frame in enumerate(results_list):
        video_writer.write_frame(frame)
    video_writer.close()


if __name__ == '__main__':
    main('./datasets/person.mp4', './result/vibe.mp4')