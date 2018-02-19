# Will change this later but this is an example of how data can be consumed.
def parse_data_point(line):
    line = line.strip()
    user, temp = line.split(":")
    ratings = temp.split(" ")
    ratings = [tuple(r.split(",")) for r in ratings]
    return user, ratings

with open("data/netflix/output") as f:
    for line in f:
        print(parse_data_point(line))

        exit()