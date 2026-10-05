import matplotlib.pyplot as plt
import numpy as np

# check data structure and format
def print_tif_info(path):
    import rasterio

    with rasterio.open(path) as src:
        print(f"Shape: ({src.count} x {src.height} x {src.width})")
        print("Channel types:", src.dtypes)
        
        sample_image = src.read()
        print(f"Distribution = min:{sample_image.min()}, mean:{round(np.mean(sample_image))}, max:{sample_image.max()}")

def convert_tif_to_np(path):
    import rasterio
    with rasterio.open(path) as src:  return src.read()

# visualize bands
def visualize_bands(image):
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))

    for i, ax in enumerate(axes.flat):
        if (i < len(image)): ax.imshow(image[i], cmap="gray")
        ax.set_title(f"Band {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_band_distributions(image):
    band_count = len(image)
    fig, axes = plt.subplots(band_count, 1, figsize=(10, 3 * band_count))

    # Make axes iterable when there is only one band
    if band_count == 1: axes = [axes]

    for i, ax in enumerate(axes):
        band = image[i].ravel()

        ax.hist(band, bins=100)
        ax.set_title(f"Band {i + 1}")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

        p1 = np.percentile(band, 1)
        p99 = np.percentile(band, 99)
        print("band:", i, "min:", band.min(), "p1:", p1, "mean:", round(np.mean(band), 2), "p99:", p99, "max:", band.max())

        ax.axvline(p1, linestyle="--")
        ax.axvline(p99, linestyle="--")

    plt.tight_layout()
    plt.show()

def percentile_normalization(image, lower_percentile, higher_percentile):
    """
    image should be shape: (channels, height, width)
    """
    image = image.astype(np.float32)

    normalized_image = np.empty_like(image)
    for i in range(image.shape[0]):
        p_lower = np.percentile(image[i], lower_percentile)
        p_higher = np.percentile(image[i], higher_percentile)

        clipped = np.clip(image[i], p_lower, p_higher)
        normalized_image[i] = ((clipped - p_lower) /(p_higher - p_lower))
    
    return normalized_image

def print_output_class_distribution(mask):
    slum_pixels = np.sum(mask == 1)
    non_slum_pixels = np.sum(mask == 0)
    total = slum_pixels + non_slum_pixels

    percentage_slums = round(100 * slum_pixels / total)
    percentage_non_slums = round(100 * non_slum_pixels / total)

    print(f"Distribution (slum : non_slum): {percentage_slums}% : {percentage_non_slums}%")

def get_bbox(mask):
    rows, cols = np.where(mask > 0)

    start_x = cols.min()
    start_y = rows.min()

    end_x = cols.max()
    end_y = rows.max()

    width = end_x - start_x + 1
    height = end_y - start_y + 1

    return int(start_x), int(start_y), int(width), int(height)

def remove_truth_area(mask, cutoff):
    mask = mask.copy()
    height, width = mask.shape
    cutoff_x = int(width * cutoff)
    mask[:, cutoff_x:] = 0
    return mask

def visualize_grayscale(images, titles):
    # visualize
    fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
    idx = 0

    for image in images:
        axes[idx].imshow(images[idx], cmap="gray")
        axes[idx].set_title(titles[idx])
        axes[idx].axis("off")
        idx += 1

    plt.tight_layout()
    plt.show()

def evaluate_metrics(y_true, y_pred):
    import pandas as pd

    metrics_report = {}
    
    # compute metrics for each class individually
    for c in [0, 1]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics_report[f"Class {c}"] = {
            "IoU (Jaccard)": f"{round(100 * iou, 2)}%",
            "Precision": f"{round(100 * precision, 2)}%",
            "Recall": f"{round(100 * recall, 2)}%",
            "F1-Score": f"{round(100 * f1, 2)}%",
        }
        
    return pd.DataFrame(metrics_report).T

def visualize_image_and_mask(image, mask):
    # apply 1st percentile to 99th percentile clipping
    normalized_image = percentile_normalization(image, 1, 99)

    # average the RGB channels
    visual_image = np.mean(normalized_image[:3], axis=0)

    # normalize mask
    if (len(mask.shape) == 3): mask = mask.astype(np.float32)[0]

    mask_min = mask.min()
    mask_max = mask.max()

    if mask_max > mask_min: normalized_mask = ((mask - mask_min) / (mask_max - mask_min))
    else: normalized_mask = np.zeros_like(mask)

    visualize_grayscale([visual_image, normalized_mask], ["input image", "ground truth"])

def visualize_prediction_custom_threshold(model, image, mask, p1, p99, threshold=0.5):
    channels, height, width = image.shape

    image_flattened = image.reshape(channels, -1).T
    image_flattened = (image_flattened - p1) / (p99 - p1)
    image_flattened = np.clip(image_flattened, 0, 1)
    
    probabilities = model.predict_proba(image_flattened)[:, 1]
    prediction = (probabilities >= threshold).astype(np.uint8)
    prediction_reshaped = prediction.reshape(height, width)

    if (threshold == 0.5): visualize_grayscale([mask, prediction_reshaped], ["input image", "prediction"])
    else: visualize_grayscale([mask, prediction_reshaped], ["input image", f"prediction (threshold = {threshold})"])

def visualize_prediction(model, image, mask, p1, p99):
    visualize_prediction_custom_threshold(model, image, mask, p1, p99)

def compare_custom_threshold_predictions(model, image, mask, p1, p99, threshold1, threshold2):
    channels, height, width = image.shape

    image_flattened = image.reshape(channels, -1).T
    image_flattened = (image_flattened - p1) / (p99 - p1)
    image_flattened = np.clip(image_flattened, 0, 1)

    probabilities1 = model.predict_proba(image_flattened)[:, 1]
    prediction1 = (probabilities1 >= threshold1).astype(np.uint8)
    prediction_reshaped1 = prediction1.reshape(height, width)

    probabilities2 = model.predict_proba(image_flattened)[:, 1]
    prediction2 = (probabilities2 >= threshold2).astype(np.uint8)
    prediction_reshaped2 = prediction2.reshape(height, width)

    visualize_grayscale(
        [mask, prediction_reshaped1, prediction_reshaped2], 
        ["input image", f"prediction1 (threshold = {threshold1})", f"prediction2 (threshold = {threshold2})"]
    )

def pad_to_multiple_of(image, multiple):
    """
    Reflect-pad an image and identify every original pixel that
    participates in the reflected padding.

    Returns
    -------
    image_padded : np.ndarray
        Reflect-padded image.
        Shape: (padded_height, padded_width)

    reflection_mask : np.ndarray
        Shape: (padded_height, padded_width)

        True  = this original pixel is used somewhere in the
                reflected padding.

        False = this original pixel is never used for padding.
    """

    _, height, width = image.shape

    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple

    padding = (
        (0, 0), # no padding on channel axis 
        (0, pad_height), # bottom padding on y-axis
        (0, pad_width) # right padding on x_axis
    )

    image_padded = np.pad(image, padding, mode="reflect")

    # np.arange creates an array containing numbers from 0 to the specified argument
    rows = np.arange(height)
    cols = np.arange(width)

    pixel_idx = (rows[:, None] * width) + cols[None, :]

    # apply the same reflection operation to the indices
    padded_ids = np.pad(
        pixel_idx,
        (
            (0, pad_height), # bottom padding on y-axis
            (0, pad_width) # right padding on x_axis
        ),
        mode="reflect"
    )

    flat_ids = padded_ids.ravel()

    # get the unique values and their total occurrence counts
    vals, counts = np.unique(flat_ids, return_counts=True)

    # original pixels that never appear again in the padded image
    # i.e. were not used to generate reflected padding.
    unique_vals = vals[counts == 1]

    # create a mask checking if elements in padded_ids match any of those unique values
    # invert mask to get non unique values i.e. reflected pixel ids
    non_unique_mask = np.isin(flat_ids, unique_vals, invert=True)
    reflection_mask = non_unique_mask.reshape(padded_ids.shape)

    return image_padded, reflection_mask

def image_to_patches(image, patch_size):

    image_padded, reflection_mask = pad_to_multiple_of(image, multiple=patch_size)
    channels, height, width = image_padded.shape

    assert height % patch_size == 0
    assert width % patch_size == 0

    n_rows = height // patch_size
    n_cols = width // patch_size

    # extract image patches
    patches = (
        image_padded.reshape(channels, n_rows, patch_size, n_cols, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, channels, patch_size, patch_size)
    )

    # extract mask patches
    patch_masks = (
        reflection_mask.reshape(n_rows, patch_size, n_cols, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch_size, patch_size)
    )

    # create boolean index for patches containing pixels that were involved in reflection padding
    # checks across spatial dimensions (-1, -2) if ANY pixel is True
    has_reflection_pixels = patch_masks.any(axis=(-1, -2))

    # separate patches into reflection and other based on if they contain pixels that were used in reflection padding
    reflection_patches = patches[has_reflection_pixels]
    other_patches = patches[~has_reflection_pixels]

    return reflection_patches, other_patches

def add_filler_bands(patches):
    image = patches[:, :10, :, :]
    mask = patches[:, -1:, :, :]

    # Create zero-filled bands
    filler_shape = (patches.shape[0], 1, patches.shape[2], patches.shape[3])

    B1  = np.zeros(filler_shape, dtype=patches.dtype)
    B9  = np.zeros(filler_shape, dtype=patches.dtype)
    B10 = np.zeros(filler_shape, dtype=patches.dtype)

    # Existing bands
    B2  = image[:, 0:1, :, :]
    B3  = image[:, 1:2, :, :]
    B4  = image[:, 2:3, :, :]
    B5  = image[:, 3:4, :, :]
    B6  = image[:, 4:5, :, :]
    B7  = image[:, 5:6, :, :]
    B8  = image[:, 6:7, :, :]
    B8A = image[:, 7:8, :, :]
    B11 = image[:, 8:9, :, :]
    B12 = image[:, 9:10, :, :]

    # Construct the final Sentinel-2 channel order
    return np.concatenate([B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12, mask], axis=1)

def assign_stratification_labels(patches):
    mask = patches[:, -1, :, :]

    # fraction of pixels belonging to class 1 for each patch
    class1_fraction = np.mean(mask == 1, axis=(1, 2))

    # assign stratification labels
    return np.digitize(
        class1_fraction,
        bins=[0.25, 0.50, 0.75],
        right=False
    )

# 70/15/15 stratified split
def split_patches(reflection_patches, other_patches):
    from sklearn.model_selection import train_test_split
    
    n_reflection_patches = len(reflection_patches)
    n_other_patches = len(other_patches)
    total_patches = n_reflection_patches + n_other_patches

    total_train_patches = int(0.7 * total_patches)
    total_test_and_val_patches = total_patches - total_train_patches
    n_train_patches_from_other_patches = total_train_patches - n_reflection_patches

    reflection_patches_labels = assign_stratification_labels(reflection_patches)
    other_patches_labels = assign_stratification_labels(other_patches)

    other_other_patches, other_train_patches, other_other_labels, other_train_labels = train_test_split(
        other_patches,
        other_patches_labels,
        test_size=n_train_patches_from_other_patches,
        random_state=42,
        stratify=other_patches_labels
    )

    train_patches = np.concatenate([reflection_patches, other_train_patches], axis=0)

    val_patches, test_patches, val_labels, test_labels = train_test_split(
        other_other_patches,
        other_other_labels,
        test_size=0.5,
        random_state=42,
        stratify=other_other_labels
    )

    return train_patches, val_patches, test_patches
