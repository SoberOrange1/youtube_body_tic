import random

def k_fold_split(video_ids, k=5, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create k random folds with specified train/val/test ratios (default 60/20/20).
    Each fold is an independent random split (not classic mutually exclusive k-fold test sets),
    increasing variability of validation selection.
    Args:
        video_ids (list): List of video ID strings.
        k (int): Number of folds.
        train_ratio (float): Proportion of videos for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.
        seed (int): Base random seed (fold index offsets this seed).
    Returns:
        list of tuples: [(train_videos, val_videos, test_videos), ...] length k.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    n = len(video_ids)
    if n < 3:
        raise ValueError("Need at least 3 videos to split into train/val/test")

    folds = []
    for i in range(k):
        # Per-fold deterministic RNG for reproducibility + variability
        rng = random.Random(seed + i)
        perm = video_ids[:]
        rng.shuffle(perm)

        test_size = max(1, int(round(n * test_ratio)))
        val_size = max(1, int(round(n * val_ratio)))
        # Ensure we do not exceed total
        if test_size + val_size >= n:
            # Reduce val_size to leave at least one for train
            val_size = max(1, n - test_size - 1)
        train_size = n - test_size - val_size

        test = perm[:test_size]
        val = perm[test_size:test_size + val_size]
        train = perm[test_size + val_size:]

        folds.append((train, val, test))

    return folds

# Example usage
if __name__ == "__main__":
    video_ids = ['01', '04', '05', '06', '07', '08', '10', '11', '14', '16', '17']
    folds = k_fold_split(video_ids, k=5, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    for idx, (train, val, test) in enumerate(folds, 1):
        print(f"Fold {idx}")
        print("Train:", train)
        print("Val:", val)
        print("Test:", test)
        print("-" * 40)
