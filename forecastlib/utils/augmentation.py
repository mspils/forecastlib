"""Time series data augmentation techniques.

This module provides various augmentation methods for enhancing time series datasets.
Implementations include simple augmentations (jitter, scaling, rotation) and more
sophisticated methods based on Dynamic Time Warping (DTW).

Augmentation can help improve model robustness and prevent overfitting on small datasets.
"""

from typing import Any

import numpy as np


def jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise (jitter) to time series data.

    Applies random Gaussian noise to each point in the series to simulate
    measurement noise and improve model robustness.

    Reference:
        https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        sigma: Standard deviation of Gaussian noise. Defaults to 0.03.

    Returns:
        Augmented array with added Gaussian noise, same shape as input.

    """
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Scale time series by random factors per feature and sample.

    Applies random scaling factors to each feature independently, simulating
    changes in amplitude or measurement scale.

    Reference:
        https://arxiv.org/pdf/1706.00527.pdf

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        sigma: Standard deviation of scaling factors (normal distribution around 1.0).
            Defaults to 0.1.

    Returns:
        Scaled array with same shape as input.

    """
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x: np.ndarray) -> np.ndarray:
    """Randomly rotate and flip features in time series data.

    Randomly permutes features and flips their signs independently for each
    sample, creating alternative feature orderings.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).

    Returns:
        Transformed array with rotated/flipped features, same shape as input.

    """
    x = np.array(x)
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x: np.ndarray, max_segments: int = 5, seg_mode: str = "equal") -> np.ndarray:
    """Permute segments of the time series along the time axis.

    Divides each time series into segments and randomly reorders them,
    creating temporal variations while preserving local patterns.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        max_segments: Maximum number of segments to divide the sequence into.
            Defaults to 5.
        seg_mode: Segmentation mode - "equal" for equal-sized segments or
            "random" for randomly sized segments. Defaults to "equal".

    Returns:
        Permuted array with same shape as input.

    """
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """Apply magnitude warping using spline interpolation.

    Uses cubic spline interpolation to create smooth amplitude variations
    across the time series, simulating changes in signal strength.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        sigma: Standard deviation of random warp control points. Defaults to 0.2.
        knot: Number of knots (control points) for spline interpolation. Defaults to 4.

    Returns:
        Magnitude-warped array with same shape as input.

    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]
        ).T
        ret[i] = pat * warper

    return ret


def time_warp(x: np.ndarray, sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """Apply time warping using spline interpolation.

    Creates warped time axes for each feature using cubic splines, effectively
    stretching or compressing different temporal regions non-uniformly.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        sigma: Standard deviation of random warp control points. Defaults to 0.2.
        knot: Number of knots (control points) for spline interpolation. Defaults to 4.

    Returns:
        Time-warped array with same shape as input.

    """
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def window_slice(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
    """Slice a random window from each time series and expand it back.

    Extracts a continuous segment and interpolates it back to original length,
    creating a compressed and re-expanded version of the series.

    Reference:
        https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        reduce_ratio: Fraction of the series to keep in the window. Defaults to 0.9.

    Returns:
        Window-sliced array with same shape as input.

    """
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i] : ends[i], dim]
            ).T
    return ret


def window_warp(x: np.ndarray, window_ratio: float = 0.1, scales: list[float] | None = None) -> np.ndarray:
    """Warp a random window of the time series by different scales.

    Selects a window and scales it (stretches or compresses) randomly,
    creating temporal distortions in localized regions.

    Reference:
        https://halshs.archives-ouvertes.fr/halshs-01357973/document

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        window_ratio: Fraction of series length to use for the warping window.
            Defaults to 0.1.
        scales: List of scaling factors to choose from. Defaults to [0.5, 2.0].

    Returns:
        Window-warped array with same shape as input.

    """
    if scales is None:
        scales = [0.5, 2.0]
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[: window_starts[i], dim]
            window_seg = np.interp(
                np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])),
                window_steps,
                pat[window_starts[i] : window_ends[i], dim],
            )
            end_seg = pat[window_ends[i] :, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(
                np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1.0, num=warped.size), warped
            ).T
    return ret


def spawner(x: np.ndarray, labels: np.ndarray, sigma: float = 0.05, verbose: int = 0) -> np.ndarray:
    """Generate new samples by averaging DTW-aligned time series from same class.

    Finds same-class samples, aligns them using DTW, and creates synthetic
    samples by averaging the aligned series at intermediate alignment points.

    Reference:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        sigma: Standard deviation for final jitter application. Defaults to 0.05.
        verbose: Verbosity level (0=silent, 1=visualize). Defaults to 0.

    Returns:
        Augmented array with spawned samples, same shape as input.

    """
    from forecastlib.utils import dtw

    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.0).astype(int)
    orig_steps = np.arange(x.shape[1])
    l_max = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l_max[choices] == l_max[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            path1 = dtw.dtw(
                pat[: random_points[i]],
                random_sample[: random_points[i]],
                dtw.RETURN_PATH,
                slope_constraint="symmetric",
                window=window,
            )
            path2 = dtw.dtw(
                pat[random_points[i] :],
                random_sample[random_points[i] :],
                dtw.RETURN_PATH,
                slope_constraint="symmetric",
                window=window,
            )
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1)
            if verbose:
                _dtw_value, cost, DTW_map, path = dtw.dtw(
                    pat, random_sample, return_flag=dtw.RETURN_ALL, slope_constraint="symmetric", window=window
                )
                dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps, np.linspace(0, x.shape[1] - 1.0, num=mean.shape[0]), mean[:, dim]
                ).T
        else:
            ret[i, :] = pat
    return jitter(ret, sigma=sigma)


def wdba(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 6,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    """Weighted Dynamic Time Warping Barycenter Averaging (WDBA).

    Creates synthetic samples by computing weighted DTW-based averages
    of same-class samples, using DTW distance as weighting metric.

    Reference:
        https://ieeexplore.ieee.org/document/8215569

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        batch_size: Number of same-class samples to use for averaging. Defaults to 6.
        slope_constraint: DTW constraint - "symmetric" or "asymmetric". Defaults to "symmetric".
        use_window: Whether to use windowing constraint in DTW. Defaults to True.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        WDBA-augmented array with same shape as input.

    """
    x = np.array(x)
    from forecastlib.utils import dtw

    window = np.ceil(x.shape[1] / 10.0).astype(int) if use_window else None
    l_max = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i in range(ret.shape[0]):
        choices = np.where(l_max == l_max[i])[0]
        if choices.size > 0:
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.0
                    else:
                        dtw_matrix[p, s] = dtw.dtw(
                            prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window
                        )

            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros(medoid_pattern.shape[0])
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.0:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(
                        medoid_pattern,
                        random_prototypes[nid],
                        dtw.RETURN_PATH,
                        slope_constraint=slope_constraint,
                        window=window,
                    )
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            ret[i, :] = x[i]
    return ret


def random_guided_warp(
    x: np.ndarray,
    labels: np.ndarray,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
    dtw_type: str = "normal",
    verbose: int = 0,
) -> np.ndarray:
    """Warp time series using random same-class sample as guide via DTW.

    Aligns each sample with a randomly selected same-class sample using DTW,
    then applies the alignment path as a warping function to create variations.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        slope_constraint: DTW constraint - "symmetric" or "asymmetric". Defaults to "symmetric".
        use_window: Whether to use windowing constraint in DTW. Defaults to True.
        dtw_type: Type of DTW - "normal" for standard DTW or "shape" for shapeDTW.
            Defaults to "normal".
        verbose: Verbosity level. Defaults to 0.

    Returns:
        Randomly guided warped array, same shape as input.

    """
    from forecastlib.utils import dtw

    window = np.ceil(x.shape[1] / 10.0).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    l_max = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)
        choices = np.where(l_max[choices] == l_max[i])[0]
        if choices.size > 0:
            random_prototype = x[np.random.choice(choices)]

            if dtw_type == "shape":
                path = dtw.shape_dtw(
                    random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window
                )
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)

            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps, np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]), warped[:, dim]
                ).T
        else:
            ret[i, :] = pat
    return ret


def random_guided_warp_shape(
    x: np.ndarray, labels: np.ndarray, slope_constraint: str = "symmetric", use_window: bool = True
) -> np.ndarray:
    """Warp time series using shapeDTW as the alignment metric.

    Convenience wrapper for random_guided_warp using shape DTW alignment.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        slope_constraint: DTW constraint - "symmetric" or "asymmetric". Defaults to "symmetric".
        use_window: Whether to use windowing constraint in DTW. Defaults to True.

    Returns:
        Random guided warped array with shape DTW, same shape as input.

    """
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")


def discriminative_guided_warp(
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 6,
    slope_constraint: str = "symmetric",
    use_window: bool = True,
    dtw_type: str = "normal",
    use_variable_slice: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    """Warp time series using discriminative guidance from class information.

    Selects a same-class prototype that maximizes separation from other classes
    based on DTW distances, then uses that prototype to guide warping. Optionally
    applies variable window slicing based on warp amount.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        batch_size: Number of samples to use for prototype selection. Defaults to 6.
        slope_constraint: DTW constraint - "symmetric" or "asymmetric". Defaults to "symmetric".
        use_window: Whether to use windowing constraint in DTW. Defaults to True.
        dtw_type: Type of DTW - "normal" or "shape". Defaults to "normal".
        use_variable_slice: Whether to apply variable window slicing based on warp amount.
            Defaults to True.
        verbose: Verbosity level. Defaults to 0.

    Returns:
        Discriminatively guided warped array, same shape as input.

    """
    from forecastlib.utils import dtw

    window = np.ceil(x.shape[1] / 10.0).astype(int) if use_window else None
    orig_steps = np.arange(x.shape[1])
    max_l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(x):
        choices = np.delete(np.arange(x.shape[0]), i)

        positive = np.where(max_l[choices] == max_l[i])[0]
        negative = np.where(max_l[choices] != max_l[i])[0]

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            pos_aves = np.zeros(pos_k)
            neg_aves = np.zeros(pos_k)
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1.0 / (pos_k - 1.0)) * dtw.shape_dtw(
                                pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window
                            )
                    for _ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1.0 / neg_k) * dtw.shape_dtw(
                            pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window
                        )
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(
                    positive_prototypes[selected_id],
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1.0 / (pos_k - 1.0)) * dtw.dtw(
                                pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window
                            )
                    for _ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1.0 / neg_k) * dtw.dtw(
                            pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window
                        )
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(
                    positive_prototypes[selected_id],
                    pat,
                    dtw.RETURN_PATH,
                    slope_constraint=slope_constraint,
                    window=window,
                )

            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(
                    orig_steps, np.linspace(0, x.shape[1] - 1.0, num=warped.shape[0]), warped[:, dim]
                ).T
        else:
            ret[i, :] = pat
            warp_amount[i] = 0.0
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp)[0]
    return ret


def discriminative_guided_warp_shape(
    x: np.ndarray, labels: np.ndarray, batch_size: int = 6, slope_constraint: str = "symmetric", use_window: bool = True
) -> np.ndarray:
    """Discriminatively guided warp using shapeDTW as the alignment metric.

    Convenience wrapper for discriminative_guided_warp using shape DTW alignment.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        labels: Class labels array of shape (batch_size,) or (batch_size, num_classes).
        batch_size: Number of samples for prototype selection. Defaults to 6.
        slope_constraint: DTW constraint - "symmetric" or "asymmetric". Defaults to "symmetric".
        use_window: Whether to use windowing constraint in DTW. Defaults to True.

    Returns:
        Discriminatively guided warped array with shape DTW, same shape as input.

    """
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def run_augmentation(x: np.ndarray, y: np.ndarray, args: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Run multiple rounds of augmentation on time series data.

    Repeatedly applies augmentation transformations specified in args, stacking
    the augmented samples with the original data.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        y: Labels array of shape (batch_size,) or (batch_size, num_classes).
        args: Configuration object with augmentation parameters:
            - augmentation_ratio: Number of augmentation rounds
            - seed: Random seed for reproducibility
            - extra_tag: Additional tag for augmentation description
            - And boolean flags for each augmentation type (jitter, scaling, etc.)

    Returns:
        Tuple of (augmented_x, augmented_y, augmentation_tag_str) where tag describes
        applied augmentations. Shapes: x is (batch_size * (1 + augmentation_ratio), seq_len, features),
        y is (batch_size * (1 + augmentation_ratio), ...), tag is a string.

    """
    print(f"Augmenting {args.data}")
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print("Round %d: %s done" % (n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag
    return x_aug, y_aug, augmentation_tags


def run_augmentation_single(x: np.ndarray, y: np.ndarray, args: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Run augmentation on a single time series or batch, handling dimension conversion.

    Handles both 2D (sequence_length, num_channels) and 3D (batch_size, sequence_length,
    num_channels) inputs, converting to 3D for processing and back to 2D if needed.

    Args:
        x: Input time series of shape (sequence_length, num_features) or
            (batch_size, sequence_length, num_features).
        y: Labels array (used for discriminative augmentations).
        args: Configuration object with augmentation parameters.

    Returns:
        Tuple of (augmented_x, augmented_y, augmentation_tag_str). X maintains
        input shape format (2D or 3D as provided).

    Raises:
        ValueError: If input x has less than 2 or more than 3 dimensions.

    """
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y

    if len(x.shape) < 3:
        x_input = x[np.newaxis, :]
    elif len(x.shape) == 3:
        x_input = x
    else:
        msg = "Input must be (batch_size, sequence_length, num_channels) dimensional"
        raise ValueError(msg)

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for _n in range(args.augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if len(x.shape) < 3:
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags


def augment(x: np.ndarray, y: np.ndarray, args: Any) -> tuple[np.ndarray, str]:
    """Apply selected augmentation transformations based on configuration.

    Applies a combination of augmentation techniques (jitter, scaling, rotation,
    permutation, magnitude/time warping, and DTW-based methods) based on boolean
    flags in the args configuration.

    Args:
        x: Input time series array of shape (batch_size, sequence_length, num_features).
        y: Labels array for use in class-aware augmentations.
        args: Configuration object with boolean flags for each augmentation type:
            - jitter, scaling, rotation, permutation, randompermutation
            - magwarp, timewarp, windowslice, windowwarp
            - spawner, dtwwarp, shapedtwwarp, wdba
            - discdtw, discsdtw

    Returns:
        Tuple of (augmented_x, augmentation_tags_str) where tags describe which
        augmentations were applied. X has same shape as input.

    """
    import forecastlib.utils.augmentation as aug

    augmentation_tags = ""
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = aug.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags
