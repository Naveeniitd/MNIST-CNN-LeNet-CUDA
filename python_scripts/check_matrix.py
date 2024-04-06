def read_floats(filename):
    with open(filename, 'r') as file:
        return [float(line) for line in file]

def compare_files(file1, file2):
    floats1 = read_floats(file1)
    floats2 = read_floats(file2)
    
    if len(floats1) != len(floats2):
        return False
    
    return all(f1 == f2 for f1, f2 in zip(floats1, floats2))

# Example usage
file1 = '/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_cpu.txt'
file2 = '/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_gpu.txt'
are_same = compare_files(file1, file2)

print(f"Files are {'the same' if are_same else 'different'}.")

