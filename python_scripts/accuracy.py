import os

# Directories
labels_dir = "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/pre-proc-img/labels_batch_binary"
probs_dir = "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/stream_output_batch"  # Assuming this is the directory name where probability files are stored

# Function to read true labels from a batch file
def read_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Function to read predictions from a batch probability file
def read_predictions(file_path):
    with open(file_path, 'r') as file:
        content = file.read().split("\n\n")
    predictions = []
    for block in content:
        if block:
            top_prediction = block.split("\n")[1].split(",")[0].split(" ")[1]  # Get the class of top prediction
            predictions.append(int(top_prediction))
    return predictions

# Initialize counters
total_images = 10000
correct_predictions = 0

# Iterate through each batch
for i in range(100):
    labels_file_path = os.path.join(labels_dir, f"batch_{i}_labels.txt")
    probs_file_path = os.path.join(probs_dir, f"batch_{i}_top5.txt")

    if os.path.exists(labels_file_path) and os.path.exists(probs_file_path):
        true_labels = read_labels(labels_file_path)
        predicted_labels = read_predictions(probs_file_path)

        # Assuming the number of labels and predictions match
        correct_predictions += sum(1 for true, pred in zip(true_labels, predicted_labels) if int(true) == pred)
    else:
        print(f"Files for batch {i} are missing.")

# Calculate accuracy
accuracy = correct_predictions / total_images
print(f"Accuracy: {accuracy*100:.2f}%")
