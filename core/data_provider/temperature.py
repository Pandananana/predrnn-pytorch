import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to use iterators.begin() to rescan from the beginning of the iterators")
            return None
        # Create batch of N sequences of length L, i.e. shape = (N, L, w, h, c)
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']  # path to temperature data folder
        self.image_width = input_param['image_width']
        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, paths, mode='train'):
        '''
        Load temperature TIFF files and create overlapping sequences
        :param paths: path to temperature data folder
        :param mode: 'train' or 'test'
        :return: data array and indices for sequences
        '''
        path = paths[0]
        print(f'Begin loading temperature data from {path}')

        # Get all TIFF files and sort them
        tiff_files = sorted([f for f in os.listdir(path) if f.endswith('.tiff') or f.endswith('.tif')])
        
        if mode == 'train':
            # Use frames 0-9 for training (exclude frame 10)
            tiff_files = [f for f in tiff_files if not f.startswith('10_')]
        elif mode == 'test':
            # For testing, we'll use all frames 0-10 to create test sequences
            # This allows us to test prediction of frame 10
            pass
        else:
            raise Exception("Unexpected mode: " + mode)

        num_frames = len(tiff_files)
        print(f"Loading {num_frames} temperature frames for {mode} set")

        if num_frames == 0:
            raise Exception(f"No TIFF files found in {path}")

        # Load first image to get dimensions
        first_img_path = os.path.join(path, tiff_files[0])
        first_img = Image.open(first_img_path)
        orig_width, orig_height = first_img.size
        print(f"Original image size: {orig_width}x{orig_height}")
        print(f"Resizing to: {self.image_width}x{self.image_width}")

        # Allocate array for all frames
        data = np.empty((num_frames, self.image_width, self.image_width, 1), dtype=np.float32)

        # Load and resize all frames
        for i, tiff_file in enumerate(tiff_files):
            img_path = os.path.join(path, tiff_file)
            # Load TIFF as grayscale
            img = Image.open(img_path).convert('L')
            img_np = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1] range
            # Note: adjust normalization based on your temperature data range
            img_min, img_max = img_np.min(), img_np.max()
            if img_max > img_min:
                img_normalized = (img_np - img_min) / (img_max - img_min)
            else:
                img_normalized = img_np
            
            # Resize to target dimensions
            img_resized = cv2.resize(img_normalized, (self.image_width, self.image_width), 
                                    interpolation=cv2.INTER_AREA)
            data[i, :, :, 0] = img_resized
            
            if i == 0:
                print(f"Temperature value range - Original: [{img_min:.2f}, {img_max:.2f}], "
                      f"Normalized: [{img_normalized.min():.2f}, {img_normalized.max():.2f}]")

        # Create overlapping sequence indices
        indices = []
        for start_idx in range(num_frames - self.seq_len + 1):
            indices.append(start_idx)

        print(f"Loaded {num_frames} frames")
        print(f"Created {len(indices)} sequences of length {self.seq_len}")
        
        if mode == 'train':
            print(f"Training sequences will use frames 0-9")
        else:
            print(f"Test sequences will include frame 10 for validation")

        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

