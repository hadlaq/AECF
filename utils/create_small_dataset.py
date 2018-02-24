# Makes a new dataset of the first 'size' lines of the larger dataset
file_path = "../data/netflix/output"
new_file_path = file_path + "_small"
size = 50000
with open(file_path) as f:
    with open(new_file_path, "a") as nf:
        for line in f:
            size -= 1
            nf.write(line)
            if size == 0:
                break
