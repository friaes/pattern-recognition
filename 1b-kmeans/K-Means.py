import csv
import torch


def csv_data(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            data.append([int(val) for val in line])
    data = torch.tensor(data)
    
    labels = data[:, 0]
    pixels = data[:, 1:].to(torch.float) # need to be float for distance calculations
    return labels, pixels


def euclidean(a: torch.Tensor, b: torch.Tensor):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1))

def K_Means(k, samples: torch.Tensor):
    cluster_centers = samples[torch.multinomial(torch.ones(len(samples)), k, replacement=False)]
    
    for i in range(100):
        #distances = calculate_distances(samples, cluster_centers, euclidean)
        distances = torch.cdist(samples, cluster_centers, p=2)

        samples_nearest_c = torch.argmin(distances, dim=-1)
        clustered_data = [samples[samples_nearest_c == i] for i in range(k)]

        old_centers = cluster_centers
        cluster_centers = torch.stack([torch.mean(cluster, dim=0) for cluster in clustered_data], dim=0)
        if torch.allclose(cluster_centers, old_centers):
            break

    return clustered_data


def C_Index(data: torch.Tensor):
    all_samples = torch.cat(data, dim=0)

    Γ = 0.0 # Sum of all distances between patterns of the same cluster
    alpha = 0  # Number of pairs of patterns that belong to the same cluster
    
    for cluster in data:
        if len(cluster) > 1:
            cluster_distances = torch.pdist(cluster, p=2)
            Γ += cluster_distances.sum()
            alpha += len(cluster_distances)

    if alpha == 0: # No pairs
        return 1.0 # Worst score
    
    # Calculate all pairwise distances in the entire dataset
    all_distances = torch.pdist(all_samples, p=2)
    all_distances_sorted = torch.sort(all_distances)[0]
    
    # Sum of the alpha smallest distances in the entire dataset
    min = all_distances_sorted[:alpha].sum()

    # Sum of the alpha largest distances
    max = all_distances_sorted[-alpha:].sum()

    if (max - min) == 0: # avoid division by zero
        return 0.0 # Best score

    C_index = (Γ - min) / (max - min)
    return C_index


def Dunn_Index(data: torch.Tensor):
    k = len(data)

    # maximum distance within each cluster, i.e., cluster diameter
    max_diameters = torch.stack([torch.pdist(cluster, p=2).max() for cluster in data])

    # cluster distances (minimum distance between all cluster pairs)
    min_cluster_distances = []
    for i in range(k):
        for j in range(i + 1, k):
            if len(data[i]) > 0 and len(data[j]) > 0:  # Skip empty clusters
                dists = torch.cdist(data[i], data[j], p=2)
                min_cluster_distances.append(dists.min())

    if len(min_cluster_distances) == 0 or len(max_diameters) == 0:
        return torch.tensor(0.0)

    min_cluster_distances = torch.stack(min_cluster_distances)

    return min_cluster_distances.min() / max_diameters.max()


def calculate_distances(test_pixels, train_pixels, distance_func):
    distances = distance_func(test_pixels.unsqueeze(1), train_pixels) # [n_test, n_train] = distance
    return distances



def main():
    train_labels, train_pixels = csv_data('../MNIST/train.csv')
    # test_labels, test_pixels = csv_data('../MNIST/test.csv')
    
    # Values for K-Means
    k_values = [5, 7, 9, 10, 12, 15]

    # quality evaluation functions
    cluster_validation_methods = {
        'C-Index': (C_Index, "C-Index"),
        #'G-K': (Goodman_Kruskal_Index, "Goodman-Kruskal Index"),
        'Dunn': (Dunn_Index, "Dunn Index"),
        # 'D-B': (Davis_Bouldin_Index, "Davis-Bouldin Index"),
    }

    methods_to_run = ['C-Index', 'Dunn']

    for k in k_values:
        print(f"\n{k} CLUSTERS:")
        data = K_Means(k, train_pixels)
        
        for method_name in methods_to_run:
            method_func, method_name = cluster_validation_methods[method_name]

            # Calculate the quality score
            score = method_func(data)
            print(f"{method_name.upper()}: {score:.3f}")

if __name__ == "__main__":
    main()

