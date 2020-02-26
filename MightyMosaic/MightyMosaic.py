import numpy as np
import tqdm.auto as tqdm
import math

OVERLAP_FACTOR = 1
FILL_MODE = 'constant'
CVAL = 0

ALLOWED_FILL_MODE = ('constant', 'nearest', 'reflect')


class MightyMosaic(np.ndarray):
    def __new__(cls, shape, tile_shape, overlap_factor=OVERLAP_FACTOR, fill_mode=FILL_MODE, cval=CVAL):
        """
        Create a MightyMosaic instance

        :param shape: Number of tiles on each axis of the mosaic. If of length 3, also describe the number of channels.
        :type shape: `list` or `tuple`, should be either of length `2` or `3`.
        :param tile_shape: Number of pixels on each axis of the tile.
        :type tile_shape: `list` or `tuple`, should be of length `2`.
        :param overlap_factor: Overlapping of neighbor tiles. If iterable, the overlap can be different on the two axis.
        :type overlap_factor: `int`, `list` or `tuple`. If iterable, should be of length 2.
        :param fill_mode: Points outside the boundaries of the input are filled according to the given mode:
                            `constant`: kkkkkkkk|abcd|kkkkkkkk (cval=k)
                            `nearest`: aaaaaaaa|abcd|dddddddd
                            `reflect`: abcddcba|abcd|dcbaabcd
        :type fill_mode: `str`, one of `constant`, `nearest` or `reflect`. Optional.
        :param cval: Value used for points outside the boundaries when fill_mode = `constant`.
        :type cval: `int`. Optional.
        :return: A subclass of `np.ndarray`, with shape of length either `4` or `5`.
        :rtype: `MightyMosaic`
        """
        assert isinstance(shape, tuple) or isinstance(shape, list), \
            f'shape {shape} should be of either a "tuple" or a "list" but is is of type "{type(shape)}"'
        assert len(shape) in (2, 3), \
            f'shape {shape} is incorrect (length is {len(shape)} but should be either 2 or 3).'

        assert isinstance(tile_shape, tuple) or isinstance(tile_shape, list), \
            f'shape {tile_shape} should be of either a "tuple" or a "list" but is is of type "{type(tile_shape)}"'
        assert len(tile_shape) == 2, \
            f'tile_shape {tile_shape} is incorrect (length is {len(tile_shape)} but should be 2.'

        assert type(overlap_factor) in (int, tuple, list), \
            f'overlap_factor should be an int or a tuple/list but is {overlap_factor} of type "{type(overlap_factor)}"'
        if isinstance(overlap_factor, int):
            overlap_factor = (overlap_factor, overlap_factor)
        assert len(overlap_factor) == 2, \
            f'When a list, overlap_factor should have a length of 2, ' \
                f'but is {overlap_factor} with a length of {len(overlap_factor)}'

        assert tile_shape[0] / overlap_factor[0] == tile_shape[0] // overlap_factor[0], \
            f"The first dimension of the tile_shape {tile_shape} cannot be divided " \
                f"by the overlap_factor {overlap_factor[0]}"
        assert tile_shape[1] / overlap_factor[1] == tile_shape[1] // overlap_factor[1], \
            f"The second dimension of the tile_shape {tile_shape} cannot be divided " \
                f"by the overlap_factor {overlap_factor[1]}"

        nb_channels = shape[-1] if len(shape) == 3 else None

        mosaic_margins = -shape[0] % tile_shape[0], -shape[1] % tile_shape[1]
        tile_margins = (int((0.5 - 0.5 / overlap_factor[0]) * tile_shape[0]),
                        int((0.5 - 0.5 / overlap_factor[1]) * tile_shape[1]))
        tile_center_dims = tile_shape[0] - 2 * tile_margins[0], tile_shape[1] - 2 * tile_margins[1]
        mosaic_shape = [math.ceil((shape[0] + mosaic_margins[0]) / tile_center_dims[0]),
                        math.ceil((shape[1] + mosaic_margins[1]) / tile_center_dims[1]),
                        tile_shape[0],
                        tile_shape[1]]
        if nb_channels:
            mosaic_shape.append(nb_channels)

        array = super().__new__(cls, mosaic_shape, float, None, 0, None, None)
        # array = np.zeros(mosaic_shape)

        array.mosaic_margins = mosaic_margins
        array.tile_margins = tile_margins
        array.tile_center_dims = tile_center_dims
        array.overlap_factor = overlap_factor
        array.fill_mode = fill_mode
        array.cval = cval
        array.original_shape = shape
        return array

    def apply(self, function, progress_bar=False, batch_size=1):
        """
        Apply a function on each tile. Progress by batching the tiles.

        :param function: The function to apply on each batchs of tile.
        :type function: callable.
        :param progress_bar: choose if a progress_bar should be used.
        :type progress_bar: `bool`. Optional, default value is `False`.
        :param batch_size: Number of batchs to pass inside the function.
        :type batch_size: `int`. Optional, default value is `1`.
        :return: A subclass of `np.ndarray`, with shape of length either `4` or `5`.
        :rtype: `MightyMosaic`.
        """
        assert isinstance(progress_bar, bool), \
            f'cval {progress_bar} should be of type "bool" but is is of type "{type(progress_bar)}"'
        assert callable(function), f'function should be callable but is of type {type(function)}'
        assert isinstance(batch_size, int), f'batch_size should be of type "int" but is of type "{type(batch_size)}"'

        index = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        batch_indexes = range(math.ceil(len(index) / batch_size))
        if progress_bar:
            batch_indexes = tqdm.tqdm(batch_indexes)

        patchs = []
        for batch_index in batch_indexes:
            min_index = batch_index * batch_size
            max_index = min(min_index + batch_size, len(index) + 1)
            batch = np.array([self[i, j] for i, j in index[min_index:max_index]])
            batch = function(batch)

            for element_index, (i, j) in enumerate(index[min_index:max_index]):
                patchs.append((i, j, batch[element_index]))

        new_shape = [self.original_shape[0], self.original_shape[1]]
        if len(patchs[0][2].shape) == 3:
            new_shape.append(patchs[0][2].shape[-1])
        new_mosaic = MightyMosaic(new_shape, patchs[0][2].shape,
                                  overlap_factor=self.overlap_factor, fill_mode=self.fill_mode, cval=self.cval)
        for i, j, patch in patchs:
            new_mosaic[i, j] = patch
        return new_mosaic

    def get_fusion(self):
        """
        Fuse the mosaic.

        :return: The fusion of each tile, with respect to the central part defined but the overlapping factor.
        :rtype: `np.ndarray`
        """
        shape = [self.shape[0] * self.tile_center_dims[0],
                 self.shape[1] * self.tile_center_dims[1]]
        if len(self.shape) == 5:
            shape.append(self.shape[-1])
        array = np.zeros(shape)
        for i, j in [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]:
            i_begin = i * self.tile_center_dims[0]
            i_end = i_begin + self.tile_center_dims[0]
            j_begin = j * self.tile_center_dims[1]
            j_end = j_begin + self.tile_center_dims[1]
            array[i_begin:i_end, j_begin:j_end] = self[i, j,
                                                  self.tile_margins[0]: self.tile_margins[0] + self.tile_center_dims[0],
                                                  self.tile_margins[1]: self.tile_margins[1] + self.tile_center_dims[1]]
        array = array[:array.shape[0] - self.mosaic_margins[0], :array.shape[1] - self.mosaic_margins[1]]
        return array

    def __copy__(self):
        new_mosaic = from_array(self, (self.shape[2], self.shape[3]), overlap_factor=self.overlap_factor,
                                fill_mode=self.fill_mode, cval=self.cval)
        return new_mosaic


def from_array(array, tile_shape, overlap_factor=OVERLAP_FACTOR,
               fill_mode=FILL_MODE, cval=CVAL):
    """
    Create a instance of `MightyMosaic` from a `np.ndarray`

    :param array: The array on which to apply the tiling.
    :type array: `np.ndarray`, it's shape should be of length either `2` or `3`.
    :param tile_shape: Number of pixels on each axis of the tile.
    :type tile_shape: `list` or `tuple`, should be of length `2`.
    :param overlap_factor: Overlapping of neighbor tiles. If iterable, the overlap can be different on the two axis.
    :type overlap_factor: `int`, `list` or `tuple`. If iterable, should be of length 2.
    :param fill_mode: Points outside the boundaries of the input are filled according to the given mode:
                        `constant`: kkkkkkkk|abcd|kkkkkkkk (cval=k)
                        `nearest`: aaaaaaaa|abcd|dddddddd
                        `reflect`: abcddcba|abcd|dcbaabcd
    :type fill_mode: `str`, one of `constant`, `nearest` or `reflect`. Optional.
    :param cval: Value used for points outside the boundaries when fill_mode = `constant`.
    :type cval: `int`. Optional.
    :return: A subclass of `np.ndarray`, with shape of length either `4` or `5`.
    :rtype: `MightyMosaic`
    """
    assert isinstance(array, np.ndarray), f'array should be of type "np.ndarray" but is of type "{type(array)}"'
    assert len(array.shape) in (2, 3), \
        f'Array has incorrect shape {array.shape} is incorrect ' \
            f'(length is {len(array.shape)} but should be either 2 or 3).'
    mosaic = MightyMosaic(array.shape, tile_shape, overlap_factor=overlap_factor, fill_mode=fill_mode, cval=cval)

    if len(array.shape) == 2:
        new_array = np.zeros((array.shape[0] + mosaic.mosaic_margins[0] + 2 * mosaic.tile_margins[0],
                              array.shape[1] + mosaic.mosaic_margins[1] + 2 * mosaic.tile_margins[1]))
    else:
        new_array = np.zeros((array.shape[0] + mosaic.mosaic_margins[0] + 2 * mosaic.tile_margins[0],
                              array.shape[1] + mosaic.mosaic_margins[1] + 2 * mosaic.tile_margins[1],
                              array.shape[2]))
    new_array[mosaic.tile_margins[0]:array.shape[0] + mosaic.tile_margins[0],
    mosaic.tile_margins[1]:array.shape[1] + mosaic.tile_margins[1]] = array
    new_array = fill(new_array, fill_mode, cval=cval,
                     i_begin=mosaic.tile_margins[0], i_end=-mosaic.tile_margins[0] - mosaic.mosaic_margins[0],
                     j_begin=mosaic.tile_margins[1], j_end=-mosaic.tile_margins[1] - mosaic.mosaic_margins[1]
                     )
    for i, j in [(i, j) for i in range(mosaic.shape[0]) for j in range(mosaic.shape[1])]:
        i_begin = i * mosaic.shape[2] // mosaic.overlap_factor[0]
        j_begin = j * mosaic.shape[3] // mosaic.overlap_factor[1]
        mosaic[i][j] = new_array[i_begin:i_begin + mosaic.shape[2], j_begin:j_begin + mosaic.shape[3]]
    return mosaic


def fill(array, fill_mode, cval=CVAL, i_begin=0, i_end=-1, j_begin=0, j_end=-1):
    """
    Fill an array on the given indexes with respect to the `fill_mode`.

    :param array: Array to fill.
    :type array: `np.ndarray`.
    :param fill_mode: Points outside the boundaries of the input are filled according to the given mode:
                        `constant`: kkkkkkkk|abcd|kkkkkkkk (cval=k)
                        `nearest`: aaaaaaaa|abcd|dddddddd
                        `reflect`: abcddcba|abcd|dcbaabcd
    :type fill_mode: `str`, one of `constant`, `nearest` or `reflect`. Optional.
    :param cval: Value used for points outside the boundaries when fill_mode = `constant`.
    :type cval: `int`. Optional.
    :param i_begin: Index, on the first axis, of the real begin of the array. Index below are filled.
    :type i_begin: `int`. Optional, default value is `0`.
    :param i_end: Index, on the first axis, of the real end of the array. Index above are filled.
    :type i_end: `int`. Optional, default value is `-1`.
    :param j_begin: Index, on the second axis, of the real begin of the array. Index below are filled.
    :type j_begin: `int`. Optional, default value is `0`.
    :param j_end: Index, on the second axis, of the real end of the array. Index above are filled.
    :type j_end: `int`. Optional, default value is `-1`.
    :return:
    """
    assert fill_mode in ALLOWED_FILL_MODE, \
        f'fill_mode {fill_mode} is not allowed, should be one of {ALLOWED_FILL_MODE}'
    assert isinstance(cval, int), f'cval {cval} should be of type "int" but is of type "{type(cval)}"'
    assert isinstance(i_begin, int), f'i_begin {i_begin} should be of type but is of type "{type(i_begin)}"'
    assert isinstance(i_end, int), f'i_end {i_end} should be of type but is of type "{type(i_end)}"'
    assert isinstance(j_begin, int), f'j_begin {j_begin} should be of type but is of type "{type(j_begin)}"'
    assert isinstance(j_end, int), f'j_end {j_end} should be of type but is of type "{type(j_end)}"'
    assert i_begin % array.shape[0] < i_end % array.shape[0], \
        f"i_begin ({i_begin}) should be less than i_end ({i_end}) ({i_begin} < {i_end} is False)"
    assert j_begin % array.shape[1] < j_end % array.shape[1], \
        f"j_begin ({j_begin}) should be less than j_end ({j_end}) ({j_begin} < {j_end} is False)"
    assert -array.shape[0] < i_begin < array.shape[0], \
        f"i_begin {i_begin} is out of range (should be in ]-{array.shape[0]}, {array.shape[0]}[)"
    assert -array.shape[0] < i_end < array.shape[0], \
        f"i_end {i_end} is out of range (should be in ]-{array.shape[0]}, {array.shape[0]}[)"
    assert -array.shape[1] < j_begin < array.shape[1], \
        f"j_begin {j_begin} is out of range (should be in ]-{array.shape[1]}, {array.shape[1]}[)"
    assert -array.shape[1] < j_end < array.shape[1], \
        f"j_end {j_end} is out of range (should be in ]-{array.shape[1]}, {array.shape[1]}[)"

    if i_end > 0:
        i_end -= array.shape[0]
    if j_end > 0:
        j_end -= array.shape[1]

    if fill_mode == 'constant':
        array[:i_begin] = cval
        array[i_end:] = cval
        array[:, :j_begin] = cval
        array[:, j_end:] = cval
    if fill_mode == 'nearest':
        array[:i_begin] = array[i_begin + 1]
        array[i_end:] = array[i_end - 1]
        array[:, :j_begin] = array[:, j_begin + 1]
        array[:, j_end:] = array[:, j_end - 1]
    if fill_mode == 'reflect':
        array[:i_begin] = array[2 * i_begin:i_begin:-1]
        array[i_end:] = array[i_end:2 * i_end:-1]
        array[:, :j_begin] = array[:, 2 * j_begin:j_begin:-1]
        array[:, j_end:] = array[:, j_end:2 * j_end:-1]
    return array
