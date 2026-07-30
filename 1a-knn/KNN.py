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


def manhattan(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.abs(a - b), dim=-1)


def calculate_distances(test_pixels, train_pixels, distance_func):
    distances = distance_func(test_pixels.unsqueeze(1), train_pixels) # [n_test, n_train] = distance
    return distances


def get_predictions(distances, train_labels, k=3):
    # get the k nearest neighbors
    _, top_indices = torch.topk(distances, k=k, dim=-1, largest=False) # [n_test, k_train_indices]
    
    # get the labels of the k nearest neighbors
    top_labels = train_labels[top_indices] # [n_test, k_train_labels]
    
    # get the common label among the k neighbors, for each test sample
    test_labels_prediction, _ = torch.mode(top_labels, dim=-1) # [n_test] = predicted_label

    return test_labels_prediction



def main():
    train_labels, train_pixels = csv_data('../MNIST/train.csv')
    test_labels, test_pixels = csv_data('../MNIST/test.csv')
    
    # Values for knn
    k_values = [1, 3, 5, 10, 15]

    # distance metric functions
    distance_functions = {
        'euclidean': (euclidean, "Euclidean"),
        'manhattan': (manhattan, "Manhattan")
    }

    distances_to_run = ['euclidean', 'manhattan']
    
    for distance_name in distances_to_run:
        distance_func, distance_name = distance_functions[distance_name]
        print(f"{distance_name.upper()} DISTANCE:")

        # calculate the distances between test and train samples   
        distances = calculate_distances(test_pixels, train_pixels, distance_func)
        
        for k in k_values:
            # get the test labels predictions according to the k nearest neighbors
            test_predictions = get_predictions(distances, train_labels, k=k)

            # calculate the accuracy of predictions
            correct = (test_predictions == test_labels).sum().item()
            accuracy = correct / len(test_labels)
            print(f"- k={k}: accuracy = {accuracy * 100:.1f}%")
        print()

if __name__ == "__main__":
    main()

