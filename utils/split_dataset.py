file_path = "../data/netflix/output"
train_file_path = file_path + "_train"
test_file_path = file_path + "_test"
dev_file_path = file_path + "_dev"

max_date = "2005-12-01"

f = open(file_path)
trf = open(train_file_path, "a")
tsf = open(test_file_path, "a")
dvf = open(dev_file_path, "a")
train_size = 0
test_size = 0
dev_size = 0
point = 0
for line in f:
    point += 1
    line = line.strip()
    user, temp = line.split(":")
    ratings = temp.split(" ")
    ratings = [tuple(r.split(",")) for r in ratings]
    old_ratings = []
    new_ratings = []
    for r in ratings:
        movie, rating, date = r
        if date < max_date:
            old_ratings.append(r)
        else:
            new_ratings.append(r)
    if len(old_ratings) > 0:
        train = ""
        for r in old_ratings:
            train += r[0] + "," + r[1] + "," + r[2] + " "
        train = train.strip() + "\n"
        train_size += 1
        trf.write(user + ":" + train)
        if len(new_ratings) > 0:
            test = user + ":"
            for r in new_ratings:
                test += r[0] + "," + r[1] + "," + r[2] + " "
            test = test.strip() + ":" + train + "\n"
            if point % 2 == 0:
                test_size += 1
                tsf.write(test)
            else:
                dev_size += 1
                dvf.write(test)

print(train_size)
print(test_size)
print(dev_size)
# 49698
# 9234
# 9056

# 477412
# 86847
# 86635


f.close()
trf.close()
tsf.close()
dvf.close()
