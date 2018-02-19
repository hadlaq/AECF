# Files to look at.
base_path = "../data/netflix/"
files = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
files = [base_path + f for f in files]

users = dict()

current_movie = "0"         # movie considered at the current line
for path in files:
    with open(path) as f:
        print("Reading file ", path, "...")
        for line in f:
            tokens = line.strip().split(",")
            if len(tokens) == 3:
                # a rating line
                user, rating, date = tokens

                item = (current_movie, rating, date)
                if user not in users:
                    users[user] = [item]
                else:
                    users[user].append(item)
            else:
                # new movie line
                current_movie = line[:line.find(":")]

    print(len(users))

print("Done reading, now writing.")
ids = list(users.keys())
i = 0
with open(base_path + "output", 'a') as f:
    for user in ids:
        line = user + ":"
        for item in users[user]:
            line += item[0] + "," + item[1] + "," + item[2] + " "
        line = line.strip()
        f.write(line + '\n')
        del users[user]
        if i % 10000 == 0:
            print("remaining: ", len(users), "/", len(ids))
        i += 1
