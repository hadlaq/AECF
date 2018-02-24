from collections import Counter
import matplotlib.pyplot as plt


def plot_dates_data(counter):
    items = list(counter.items())
    items.sort()
    keys = [t[0] for t in items]
    ticks = list(keys)
    values = [t[1] for t in items]

    # print every 8 months
    for i in range(len(ticks)):
        if i % 10 != 0:
            ticks[i] = ""

    plt.bar(keys, values, color='r')
    plt.axes().get_xaxis().set_ticklabels(ticks)
    plt.title("Distribution of ratings over time")
    plt.ylabel("Number of ratings")
    plt.xlabel("Months")
    plt.show()


def plot_users_data(counter):
    ranges = [100 * i for i in range(1, 6)]
    buckets = Counter()
    for user in counter:
        ratings = counter[user]
        bucket = ranges[-1]
        for r in ranges:
            if ratings <= (r + 0.5):
                bucket = r
                break
        buckets[bucket] += 1

    ticks = list(buckets.keys())
    for i in range(len(ticks)):
        if ticks[i] < 10:
            ticks[i] = "-0" + str(ticks[i])
        else:
            ticks[i] = "-" + str(ticks[i])

    for i in range(len(ticks)):
        if ticks[i] == "-" + str(ranges[-1]):
            ticks[i] = ">= " + str(ranges[-1])
            break

    print(buckets)
    print(ticks)
    print(buckets.values())

    plt.bar(ticks, buckets.values(), color='r')
    plt.title("Number of ratings submitted per user")
    plt.ylabel("Number of users")
    plt.xlabel("Number of ratings")
    plt.show()


def plot_movies_data(counter):
    ranges = [1000, 2000, 3000, 4000, 5000, 6000]
    buckets = Counter()
    for movie in counter:
        ratings = counter[movie]
        bucket = ranges[-1]
        for r in ranges:
            if ratings <= (r + 0.5):
                bucket = r
                break
        buckets[bucket] += 1

    ticks = ["-" + str(k) for k in buckets.keys()]
    for i in range(len(ticks)):
        if ticks[i] == "-" + str(ranges[-1]):
            ticks[i] = ">= " + str(ranges[-1])
            break

    plt.bar(ticks, buckets.values(), color='r')
    plt.title("Number of ratings submitted for each movie")
    plt.ylabel("Number of movies")
    plt.xlabel("Number of ratings")
    plt.show()

# Files to look at.
base_path = "../data/netflix/"
files = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
files = [base_path + f for f in files]

# Find the first and last rating date in the dataset.
first_date = "9999-99-99"
last_date = "0000-00-00"

# Find users with most/least number of ratings
max_user = -999999
min_user = 9999999

# Find users with most/least number of ratings
max_movie = -999999
min_movie = 9999999

current_movie = "0"         # movie considered at the current line
dates_counter = Counter()   # date to number of ratings
movies_counter = Counter()  # movie to number of ratings
users_counter = Counter()   # user to number of ratings

ratings_per_file = -1    # -1 for all
for path in files:
    total_ratings = 0
    with open(path) as f:
        print("Reading file ", path, "...")
        for line in f:
            tokens = line.split(",")
            if len(tokens) == 3:
                # a rating line
                user, rating, date = tokens

                movies_counter[current_movie] += 1
                max_movie = max(max_movie, movies_counter[current_movie])
                min_movie = min(min_movie, movies_counter[current_movie])

                users_counter[user] += 1
                max_user = max(max_user, users_counter[user])
                min_user = min(min_user, users_counter[user])

                first_date = min(first_date, date)
                last_date = max(first_date, date)
                reduced_date = date[:date.rfind("-")]   # YYYY-mm-dd ==> YYYY-mm
                dates_counter[reduced_date] += 1

                total_ratings += 1
                if total_ratings == ratings_per_file:
                    break
            else:
                # new movie line
                current_movie = line[:line.find(":")]

print()
print("First date found: ", first_date)
print("Last date found: ", last_date)
plot_dates_data(dates_counter)

print()
print("User with most ratings: ", max_user)
print("User with least ratings: ", min_user)
plot_users_data(users_counter)

print()
print("Movie with most ratings: ", max_movie)
print("Movie with least ratings: ", min_movie)
plot_movies_data(movies_counter)
