import multiprocessing
import numpy as np
import pandas as pd
from numba import prange

class RocketConfigurable:
    """
    ROCKET (RandOm Convolutional KErnel Transform) with configurable features.
    
    Based on sktime's implementation but allows choosing which features to extract:
    - 'ppv': Proportion of Positive Values only
    - 'ppv+mean': PPV and Mean
    - 'ppv+max': PPV and Max (original ROCKET)
    
    Parameters
    ----------
    num_kernels : int, default=10000
        Number of random convolutional kernels
    features : str, default='ppv+max'
        Which features to extract. Options: 'ppv', 'ppv+mean', 'ppv+max'
    normalise : bool, default=True
        Whether to normalize input time series per instance
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all processors
    random_state : int or None, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, num_kernels=10000, features='ppv+max', 
                 normalise=True, random_state=None):
        self.num_kernels = num_kernels
        self.features = features
        self.normalise = normalise
        self.random_state = random_state if isinstance(random_state, int) else None
        self.kernels = None
        self.n_columns = None
        
        # Validate features parameter
        valid_features = ['ppv', 'ppv+mean', 'ppv+max']
        if features not in valid_features:
            raise ValueError(f"features must be one of {valid_features}, got '{features}'")
    
    def fit(self, X, y=None):
        """
        Generate random kernels adjusted to time series shape.
        
        Parameters
        ----------
        X : array-like, shape (n_instances, n_channels, n_timesteps)
            Panel of time series to transform
        y : ignored
            
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        # Number of columns correspond to number of channels/dimensions in our timeseries
        # Number of timepoints is the number of timesteps (must be the same across the inputs)
        _, self.n_columns, n_timepoints = X.shape
        
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, self.n_columns, self.random_state
        )
        return self
    
    def transform(self, X, y=None):
        """
        Transform input time series using random convolutional kernels.
        
        Parameters
        ----------
        X : array-like, shape (n_instances, n_channels, n_timesteps)
            Panel of time series to transform
        y : ignored
            
        Returns
        -------
        DataFrame of transformed features
        """
        X = np.asarray(X, dtype=np.float32)
        
        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        
        # Apply kernels -> always get ppv, mean, max
        full_features = _apply_kernels_ppv_mean_max(X, self.kernels)
        
        # Using slicing to get features that are requested
        ppv = full_features[:, 0::3]
        mean = full_features[:, 1::3]
        max_ = full_features[:, 2::3]

        if self.features == 'ppv':
            result = ppv
        elif self.features == 'ppv+mean':
            result = np.empty((full_features.shape[0], self.num_kernels * 2), dtype=full_features.dtype)
            result[:, 0::2] = ppv
            result[:, 1::2] = mean
        else:  # ppv+max, the original 
            result = np.empty((full_features.shape[0], self.num_kernels * 2), dtype=full_features.dtype)
            result[:, 0::2] = ppv
            result[:, 1::2] = max_
        
        return pd.DataFrame(result)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)


def _generate_kernels(n_timepoints, num_kernels, n_columns, seed):
    """Generate random kernel parameters."""
    if seed is not None:
        np.random.seed(seed)

    # Each length has an equal probability of being chosen; length selection is random
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels).astype(np.int32)

    # Deciding the number of channels that will be "looked by" the kernel
    # Is limited by the kernels own length
    num_channel_indices = np.zeros(num_kernels, dtype=np.int32)
    for i in range(num_kernels):
        limit = min(n_columns, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    # This is a flat array which will store specific channels each kernel will use
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    # Allocating space for weights for each channel in each kernel
    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(num_kernels):
        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        # The weights are sampled from the normal distribution and are mean centered (line 149)
        _weights = np.random.normal(0, 1, _num_channel_indices * _length).astype(
            np.float32
        )

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0
        # Here, mean weight is computed per each channels and is subtracted from the original weight
        # Essentially, our kernels could be stored as 2D (channels x length), but the weights are instead stored in 1D-
        # They will be transformed later into 2D during application

        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        # Storing the actual indices of channels that the kernel will interct with
        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_columns), _num_channel_indices, replace=False
        )

        # Bias is sampled from the uniform distribution between -1 and 1. 
        # It is added after the convolution operation between an input and the kernel weights of the  kernel
        # Bias introduces a difference between two potentially simialr kernels by changing values in the
        # feature maps, so different feature aspects might be captured 
        biases[i] = np.random.uniform(-1, 1)

        # Dilation controls the spacing between the kernel weights when you apply convolution. 
        # Dilation is sampled as follows: d = [2^x], where x is sampled from uniform distribution
        # (0, A) where A = log2((input_length-1)/(kernel_length-1)). 
        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        # Padding is necessary so that the kernel is centered at the first and last point of the series.
        # In ROCKET, it whether padding will be used per kernel is decided randomly.
        # If yes, then N number of 0s will be appended at the end and start of the series, where
        # N is ((kernel_length-1)*dilation)/2
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )

def _apply_kernel_multivariate_all(
    X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices
):
    """Apply one multivariate kernel on one sample and return PPV, mean, and max."""
    n_columns, n_timepoints = X.shape
    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)
    
    _ppv = 0
    _max = -np.inf
    _sum_all = 0.0
    
    end = (n_timepoints + padding) - ((length - 1) * dilation)
    
    # We are looping the kernel across time series, with i being a different starting position
    for i in range(-padding, end):
        _sum = bias # adding bias already
        index = i
        
        # This is simply a kernel application. We compute feature map per channel and then sum it up. 
        for j in range(length):
            if index > -1 and index < n_timepoints:
                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]
            index = index + dilation
        
        _sum_all += _sum
        
        # If an individual value is bigger than current max, store it
        if _sum > _max:
            _max = _sum
        
        # If an inidividual value is positive, increment the count of positive values
        if _sum > 0:
            _ppv += 1
    
    ppv = np.float32(_ppv / output_length)
    mean = np.float32(_sum_all / output_length)
    max_val = np.float32(_max)
    
    return ppv, mean, max_val


def _apply_kernels_ppv_mean_max(X, kernels):
    """Apply kernels and extract PPV + mean + max"""
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    ) = kernels

    n_instances, n_columns, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((n_instances, num_kernels * 3), dtype=np.float32)


    # Looping through each sample
    for i in prange(n_instances):

        # The same principle as when genereting kernels
        a1 = 0
        a2 = 0
        a3 = 0

        for j in range(num_kernels):
            b1 = a1 + num_channel_indices[j] * lengths[j] #
            b2 = a2 + num_channel_indices[j]
            b3 = a3 + 3
            _weights = weights[a1:b1].reshape((num_channel_indices[j], lengths[j]))
            ppv, mean, max_val = _apply_kernel_multivariate_all(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    num_channel_indices[j],
                    channel_indices[a2:b2],
                )
            _X[i, a3:b3] = (ppv, mean, max_val)

            a1 = b1
            a2 = b2
            a3 = b3

    return _X