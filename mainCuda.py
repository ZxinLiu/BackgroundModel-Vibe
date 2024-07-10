from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from src.utils import mkdirs
from src.ViBe import ViBe
import numpy as np
import cupy as cp
import tqdm
import time
import cv2


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

    gpu_mat = cv2.cuda_GpuMat()
    idx = 0
    time_interval = 1
    for i in tqdm.tqdm(range(0, frame_count, 1)):
        if i % time_interval == 0:
            ret = video_cap.grab()
            if not ret:
                break
            
            ret, frame = video_cap.retrieve()
            if frame is None:
                break

            base_frame = frame.copy()
            gpu_mat.upload(frame)
            # opencv numpy.ndarray -> cv2.cuda_GpuMat
            frame = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_RGB2GRAY)  
            frame = cv2.cuda.resize(frame, (re_width, re_height))
            # cv2.cuda_GpuMat -> numpy.ndarray
            frame = frame.download()  

            # the first frame
            if i == 0:
                vibe.processFirstFrame(frame)
                cp_binary, white_count = vibe.getBGmodel()
            else:
                vibe.updateBGmodel(frame)
                cp_binary, white_count = vibe.getBGmodel()

            np_binary = cp.asnumpy(cp_binary)
            np_binary = np.tile(np_binary[..., np.newaxis], (1, 1, 3))
            img_merge = np.concatenate((frame, np_binary), axis=1)
            results_list.append(img_merge)

            idx += 1

    # initialize video parameters
    frame_size = (re_width * 2, re_height) 
    fps = int(fps)

    # create the video writer
    video_writer = FFMPEG_VideoWriter(output_path, fps=fps, size=frame_size)
    for i, frame in enumerate(results_list):
        video_writer.write_frame(frame)
    video_writer.close()


if __name__ == '__main__':
    main('./datasets/person.mp4', './vibe.mp4')