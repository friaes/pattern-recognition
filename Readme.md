# Pattern Recognition

Individual exercises for the **Pattern Recognition** course at the **University of Fribourg** (BeNeFri joint MSc programme), Fall 2025. Each task implements a core pattern-recognition algorithm **from scratch in PyTorch** on the **MNIST** handwritten-digit dataset (each sample is a class label plus 784 greyscale pixel values, used directly as features).

| Exercise | Task | Type |
|----------|------|------|
| 1a | k-Nearest Neighbours classification | code |
| 1b | K-Means clustering + cluster-validity indices | code |
| 2 | Convolutional Neural Network (forward/backprop by hand) | pen-and-paper |

> The **group** programming component of Exercise 2 — a comparison of SVM, MLP and CNN classifiers on MNIST — lives in its own repository, [`mnist-classifiers`](https://github.com/friaes/mnist-classifiers)

## Exercise 1a — KNN classification

A k-nearest-neighbours classifier (`1a-knn/KNN.py`), written with fully **vectorised** tensor operations (no per-sample Python loops):

- Loads the CSV data into tensors (labels + float pixel features).
- Computes an `[n_test, n_train]` distance matrix by broadcasting, for two metrics: **Euclidean** and **Manhattan**.
- Selects the `k` nearest training samples with `torch.topk` and predicts each test label by majority vote (`torch.mode`).
- Sweeps `k in {1, 3, 5, 10, 15}` for both metrics and reports test accuracy.

**Result (`1a-knn/reportKNN.md`):** Euclidean distance beat Manhattan at every `k`, with the best accuracy (~88.6%) at `k = 1`, decreasing as `k` grew.

## Exercise 1b — K-Means clustering

A K-Means implementation (`1b-kmeans/K-Means.py`) applied to the MNIST training set, with cluster quality assessed by two **internal validation indices implemented from scratch**:

- **K-Means**: random centroid initialisation (`torch.multinomial`), then iterate assignment (`torch.cdist` + `argmin`) and centroid update (per-cluster mean) until convergence or 100 iterations.
- **C-Index**: compares the sum of within-cluster pairwise distances against the theoretical best/worst case over all pairwise distances (lower is better).
- **Dunn Index**: ratio of the minimum inter-cluster distance to the maximum cluster diameter (higher is better).
- Sweeps `k in {5, 7, 9, 10, 12, 15}` and reports both indices.

**Result (`1b-kmeans/K-Means.md`):** the indices disagree on the optimal `k` — C-Index favours `k = 10` (matching the ten digit classes), the Dunn Index favours `k = 9`/`7`.

## Exercise 2 — CNN (pen-and-paper)

The individual part of Exercise 2 is a **hand-worked exercise** computing the operations of a small convolutional neural network by hand (convolution / forward pass and the associated backpropagation), submitted as a scanned document: `2-cnn-individual/exercise2_Rodrigo_Friaes.pdf`. There is no code for this part.

## Data

The MNIST CSVs (`train.csv`, `test.csv`) are **not included**. Each script expects them at `../MNIST/` relative to itself. Each row is `label, pixel_0, ..., pixel_783` with pixel values in `[0, 255]`.

## Requirements

- Python 3
- PyTorch (`pip install torch`)

## Usage

```bash
# KNN classification
cd 1a-knn && python3 KNN.py

# K-Means clustering
cd 1b-kmeans && python3 K-Means.py
```

Both scripts print their results to the console (also captured in the accompanying `.md` reports).

## Repository layout

```
.
├── 1a-knn/
│   ├── KNN.py
│   ├── reportKNN.md               # results + short discussion
│   └── Exercise-1a.pdf            # assignment brief
├── 1b-kmeans/
│   ├── K-Means.py
│   ├── K-Means.md                 # results + short discussion
│   └── Exercise-1b.pdf            # assignment brief
└── 2-cnn-individual/
    ├── exercise2_Rodrigo_Friaes.pdf   # hand-worked CNN exercise
    └── Exercise-2-CNN-individual-Solution.pdf
```

## Notes

The algorithms (KNN, K-Means, and the C-Index / Dunn Index cluster-validity measures) are implemented from scratch rather than called from a library. Master's-level coursework at the University of Fribourg.
