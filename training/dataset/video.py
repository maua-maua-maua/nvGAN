import copy
import json
import os
import random
import zipfile
from typing import Dict, List, Tuple, Union

import numpy as np
import PIL.Image
import torch

import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

from . import Dataset

NUMPY_INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64, np.single, np.double]


class VideoFramesFolderDataset(Dataset):
    def __init__(self,
        path,                                           # Path to directory or zip.
        resolution=None,                                # Unused arg for backward compatibility
        load_n_consecutive: int=None,                   # Should we load first N frames for each video?
        load_n_consecutive_random_offset: bool=True,    # Should we use a random offset when loading consecutive frames?
        subsample_factor: int=1,                        # Sampling factor, i.e. decreasing the temporal resolution
        discard_short_videos: bool=False,               # Should we discard videos that are shorter than `load_n_consecutive`?
        **super_kwargs,                                 # Additional arguments for the Dataset base class.
    ):  
        self.max_num_frames = 1024
        self.sampling_kwargs = dict(
            type="random",
            dists=[1, 2, 4, 8, 16, 32],
            num_frames_per_sample=2,
        )
        self._path = path
        self._zipfile = None
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        self.discard_short_videos = discard_short_videos

        if self.subsample_factor > 1 and self.load_n_consecutive is None:
            raise NotImplementedError("Can do subsampling only when loading consecutive frames.")

        listdir_full_paths = lambda d: sorted([os.path.join(d, x) for x in os.listdir(d)])
        name = os.path.splitext(os.path.basename(self._path))[0]

        if os.path.isdir(self._path):
            self._type = 'dir'
            # We assume that the depth is 2
            self._all_objects = {o for d in listdir_full_paths(self._path) for o in (([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._all_objects = {os.path.relpath(o, start=os.path.dirname(self._path)) for o in {self._path}.union(self._all_objects)}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_objects = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must be either a directory or point to a zip archive')

        PIL.Image.init()
        self._video_dir2frames = {}
        objects = sorted([d for d in self._all_objects])
        root_path_depth = len(os.path.normpath(objects[0]).split(os.path.sep))
        curr_d = objects[1] # Root path is the first element

        for o in objects[1:]:
            curr_obj_depth = len(os.path.normpath(o).split(os.path.sep))

            if self._file_ext(o) in PIL.Image.EXTENSION:
                assert o.startswith(curr_d), f"Object {o} is out of sync. It should lie inside {curr_d}"
                assert curr_obj_depth == root_path_depth + 2, "Frame images should be inside directories"
                if not curr_d in self._video_dir2frames:
                    self._video_dir2frames[curr_d] = []
                self._video_dir2frames[curr_d].append(o)
            elif self._file_ext(o) == 'json':
                assert curr_obj_depth == root_path_depth + 1, "Classes info file should be inside the root dir"
                pass
            else:
                # We encountered a new directory
                assert curr_obj_depth == root_path_depth + 1, f"Video directories should be inside the root dir. {o} is not."
                if curr_d in self._video_dir2frames:
                    sorted_files = sorted(self._video_dir2frames[curr_d])
                    if sorted_files != self._video_dir2frames[curr_d]:
                        print('BAD ORDER!')
                        for f in self._video_dir2frames[curr_d]:
                            print('-', f)
                        assert False
                    self._video_dir2frames[curr_d] = sorted_files
                curr_d = o

        if self.discard_short_videos:
            self._video_dir2frames = {d: fs for d, fs in self._video_dir2frames.items() if len(fs) >= self.load_n_consecutive * self.subsample_factor}

        self._video_idx2frames = [frames for frames in self._video_dir2frames.values()]

        if len(self._video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        raw_shape = [len(self._video_idx2frames)] + list(self._load_raw_frames(0)[0][0].shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(os.path.dirname(self._path), fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_labels(self):
        """
        We leave the `dataset.json` file in the same format as in the original SG2-ADA repo:
        it's `labels` field is a hashmap of filename-label pairs.
        """
        fname = 'dataset.json'
        labels_files = [f for f in self._all_objects if f.endswith(fname)]
        if len(labels_files) == 0:
            return None
        assert len(labels_files) == 1, f"There can be only a single {fname} file"
        with self._open_file(labels_files[0]) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None

        labels = dict(labels)
        # The `dataset.json` file defines a label for each image and
        # For the video dataset, this is both inconvenient and redundant.
        # So let's redefine this
        video_labels = {}
        for filename, label in labels.items():
            dirname = os.path.dirname(filename)
            if dirname in video_labels:
                assert video_labels[dirname] == label
            else:
                video_labels[dirname] = label
        labels = video_labels
        labels = [labels[os.path.normpath(dname).split(os.path.sep)[-1]] for dname in self._video_dir2frames]
        labels = np.array(labels)

        if labels.dtype in NUMPY_INTEGER_TYPES:
            labels = labels.astype(np.int64)
        elif labels.dtype in NUMPY_FLOAT_TYPES:
            labels = labels.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels.dtype}")

        return labels

    def __getitem__(self, idx: int) -> Dict:
        if self.load_n_consecutive:
            num_frames_available = len(self._video_idx2frames[self._raw_idx[idx]])
            assert num_frames_available - self.load_n_consecutive * self.subsample_factor >= 0, f"We have only {num_frames_available} frames available, cannot load {self.load_n_consecutive} frames."

            if self.load_n_consecutive_random_offset:
                random_offset = random.randint(0, num_frames_available - self.load_n_consecutive * self.subsample_factor + self.subsample_factor - 1)
            else:
                random_offset = 0
            frames_idx = np.arange(0, self.load_n_consecutive * self.subsample_factor, self.subsample_factor) + random_offset
        else:
            frames_idx = None

        frames, times = self._load_raw_frames(self._raw_idx[idx], frames_idx=frames_idx)

        assert isinstance(frames, np.ndarray)
        assert list(frames[0].shape) == self.image_shape
        assert frames.dtype == np.uint8
        assert len(frames) == len(times)

        if self._xflip[idx]:
            assert frames.ndim == 4 # TCHW
            frames = frames[:, :, :, ::-1]

        return {
            'image': frames.copy(),
            'label': self.get_label(idx),
            'times': times,
            'video_len': self.get_video_len(idx),
        }

    def get_video_len(self, idx: int) -> int:
        return min(self.max_num_frames, len(self._video_idx2frames[self._raw_idx[idx]]))

    def _load_raw_frames(self, raw_idx: int, frames_idx: List[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        frame_paths = self._video_idx2frames[raw_idx]
        total_len = len(frame_paths)
        offset = 0
        images = []

        if frames_idx is None:
            if total_len > self.max_num_frames:
                offset = random.randint(0, total_len - self.max_num_frames)
            frames_idx = sample_frames(total_video_len=min(total_len, self.max_num_frames), **self.sampling_kwargs) + offset
        else:
            frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            with self._open_file(frame_paths[frame_idx]) as f:
                images.append(load_image_from_buffer(f))

        return np.array(images), frames_idx - offset

    def compute_max_num_frames(self) -> int:
        return max(len(frames) for frames in self._video_idx2frames)

#----------------------------------------------------------------------------

def load_image_from_buffer(f, use_pyspng: bool=False) -> np.ndarray:
    if use_pyspng:
        image = pyspng.load(f.read())
    else:
        image = np.array(PIL.Image.open(f).convert('RGB'))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    image = image.transpose(2, 0, 1) # HWC => CHW

    return image

#----------------------------------------------------------------------------

def video_to_image_dataset_kwargs(video_dataset_kwargs: dnnlib.EasyDict) -> dnnlib.EasyDict:
    """Converts video dataset kwargs to image dataset kwargs"""
    return dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=video_dataset_kwargs.path,
        use_labels=video_dataset_kwargs.use_labels,
        xflip=video_dataset_kwargs.xflip,
        resolution=video_dataset_kwargs.resolution,
        random_seed=video_dataset_kwargs.get('random_seed'),
        # Explicitly ignoring the max size, since we are now interested
        # in the number of image instead of the number of videos
        # max_size=video_dataset_kwargs.max_size,
    )

#----------------------------------------------------------------------------

def sample_frames(total_video_len: int, **kwargs) -> np.ndarray:
    if kwargs['type'] == 'random':
        return random_frame_sampling(total_video_len, **kwargs)
    elif kwargs['type'] == 'uniform':
        return uniform_frame_sampling(total_video_len, **kwargs)
    else:
        raise NotImplementedError

#----------------------------------------------------------------------------

def random_frame_sampling(total_video_len: int, num_frames_per_sample: int, dists: Union[List, Tuple], max_time_diff: int=float('inf'), max_dist: int=float('inf'), use_fractional_t: bool=False) -> np.ndarray:
    min_time_diff = num_frames_per_sample - 1
    max_time_diff = min(total_video_len - 1, max_dist, max_time_diff)

    if type(dists) in (list, tuple):
        time_diff_range = [d for d in dists if min_time_diff <= d <= max_time_diff]
    else:
        time_diff_range = range(min_time_diff, max_time_diff)

    time_diff: int = random.choice(time_diff_range)
    if use_fractional_t:
        offset = random.random() * (total_video_len - time_diff - 1)
    else:
        offset = random.randint(0, total_video_len - time_diff - 1)
    frames_idx = [offset]

    if num_frames_per_sample > 1:
        frames_idx.append(offset + time_diff)

    if num_frames_per_sample > 2:
        frames_idx.extend([(offset + t) for t in random.sample(range(1, time_diff), k=num_frames_per_sample - 2)])

    frames_idx = sorted(frames_idx)

    return np.array(frames_idx)

#----------------------------------------------------------------------------

def uniform_frame_sampling(total_video_len: int, num_frames_per_sample: int, dists: Union[List, Tuple], max_time_diff: int=float('inf'), max_dist: int=float('inf'), use_fractional_t: bool=False) -> np.ndarray:
    # Step 1: Select the distance between frames
    if type(dists) in (list, tuple):
        valid_dists = [d for d in dists if (d * num_frames_per_sample - d + 1) <= min(total_video_len, max_time_diff)]
        d = random.choice(valid_dists)
    else:
        max_dist = min(max_dist, total_video_len // num_frames_per_sample, max_time_diff // num_frames_per_sample)
        d = random.randint(1, max_dist)

    d_total = d * num_frames_per_sample - d + 1

    # Step 2: Sample.
    if use_fractional_t:
        offset = random.random() * (total_video_len - d_total)
    else:
        offset = random.randint(0, total_video_len - d_total)

    frames_idx = offset + np.arange(num_frames_per_sample) * d

    return frames_idx

#----------------------------------------------------------------------------
