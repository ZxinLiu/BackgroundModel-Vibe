import cupy as cp
# import numpy as np


class ViBe():
    def __init__(self, width, height, n_samples=20, n_min_matches=3, radius=20, random_sample=16):
        """
        Initializes the ViBe model parameters.

        Args:
            width (int): The width of the frame, which should be the same as the width of the image.
            height (int): The height of the frame, which should be the same as the height of the image.
            n_samples (int, optional): The number of samples to consider. Defaults to 20.
            n_min_matches (int, optional): The minimum number of matches required. Defaults to 3.
            radius (int, optional): The pixel radius parameter. Defaults to 20.
            random_sample (int, optional): The random sample parameter. Defaults to 16.

        Returns:
            None
        """

        self.__width = width
        self.__height = height
        self.__n_samples = n_samples
        self.__BgModel = cp.zeros((height, width))
        self.__SampleLib = cp.zeros((n_samples, height, width))
        self.__BgCount = cp.zeros((height, width))

        self.__n_min_matches = n_min_matches
        self.__radius = radius
        self.__random_sample = random_sample

        self.__foreground = 255
        self.__background = 0


    def processFirstFrame(self, frame):
        """
        Process the first frame of the video.
        
        Args:
            frame (numpy.ndarray): The first frame of the video.
        
        Returns:
            None
        """
        frame = cp.asarray(frame)
        xy_random_tensor = cp.random.randint(-1, 2, size=(2, self.__n_samples, self.__height, self.__width))
        x_idx = cp.arange(self.__width)
        x_idx_mat = cp.expand_dims(x_idx, axis=0).repeat(self.__height, axis=0)
        y_idx = cp.arange(self.__height)
        y_idx_mat = cp.expand_dims(y_idx, axis=1).repeat(self.__width, axis=1)

        xy_idx_tensor = cp.zeros((2, self.__n_samples, self.__height, self.__width))

        for k in range(self.__n_samples):
            xy_idx_tensor[0, k, :, :] = x_idx_mat
            xy_idx_tensor[1, k, :, :] = y_idx_mat

        xy_idx_tensor = xy_idx_tensor + xy_random_tensor

        xy_idx_tensor[xy_idx_tensor < 0] = 0
        # width->x
        x_idx_bb = xy_idx_tensor[0, :, :, -1]
        x_idx_bb[x_idx_bb >= self.__width] = self.__width - 1
        # height->y
        y_idx_bb = xy_idx_tensor[1, :, -1, :]
        y_idx_bb[y_idx_bb >= self.__height] = self.__height - 1
        # x, y->xy
        xy_idx_tensor[0, :, :, -1] = x_idx_bb
        xy_idx_tensor[1, :, -1, :] = y_idx_bb

        xy_idx_tensor = xy_idx_tensor.astype(int)
        
        self.__SampleLib = frame[xy_idx_tensor[1, :, :, :], xy_idx_tensor[0, :, :, :]]

    # @profile
    def updateBGmodel(self, frame):
        """
        Updates the background model based on the input frame.

        Args:
            frame: The input frame to update the background model.

        Returns:
            None
        """
        frame = cp.asarray(frame)
        dist = cp.abs((self.__SampleLib.astype(float) - frame.astype(float)).astype(int))
        dist = cp.where(dist < self.__radius, 1, 0)
        matches = cp.sum(dist, axis=0)
        matches = matches < self.__n_min_matches
        self.__BgModel = cp.where(matches, self.__foreground, self.__background)
        self.__BgCount = cp.where(matches, self.__BgCount + 1, 0)
        matches = cp.where(self.__BgCount > 50, False, matches)
        update_factor = cp.random.randint(self.__random_sample, size=(self.__height, self.__width))
        update_factor = cp.where(matches, 1, update_factor)
        update_idx = cp.where(update_factor == 0)
        update_sample_position = cp.random.randint(self.__n_samples, size=(update_idx[0].shape))
        self.__SampleLib[update_sample_position, update_idx[0], update_idx[1]] = frame[update_idx[0], update_idx[1]]
        update_factor = cp.random.randint(self.__random_sample, size=(self.__height, self.__width))
        update_factor = cp.where(matches, 1, update_factor)
        update_idx = cp.where(update_factor == 0)
        update_idx_neighbor_number = update_idx[0].shape[0]

        xy_offset_tensor = cp.random.randint(-1, 2, size=(2, update_idx_neighbor_number))
        xy_idx_tensor = cp.stack(update_idx)
        xy_neighbor_tensor = xy_idx_tensor + xy_offset_tensor

        xy_neighbor_tensor[xy_neighbor_tensor < 0] = 0
        xy_neighbor_tensor[0, xy_neighbor_tensor[0, :] >= self.__height] = self.__height - 1
        xy_neighbor_tensor[1, xy_neighbor_tensor[1, :] >= self.__width] = self.__width - 1
        update_sample_position = cp.random.randint(self.__n_samples, size=(update_idx_neighbor_number))
        self.__SampleLib[update_sample_position, xy_neighbor_tensor[0], xy_neighbor_tensor[1]] = frame[update_idx[0], update_idx[1]]

    def getBGmodel(self):
        """
        Converts the background model to an unsigned byte type and returns the model and the sum of its elements.

        Returns:
            Tuple[ndarray, int]: A tuple containing the background model as an unsigned byte array and the sum of its elements.
        """

        self.__BgModel = self.__BgModel.astype(cp.uint8)
        return self.__BgModel, int(cp.sum(cp.sum(self.__BgModel, axis=0), axis=0) / 255)
