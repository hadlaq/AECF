from collections import Counter
import matplotlib.pyplot as plt

first_date = "9999-99-99"
last_date = "0000-00-00"

base_path = "../data/netflix/"
files = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
files = [base_path + f for f in files]

movie = "0"
dates_counter = Counter()
movies_counter = Counter()
users_counter = Counter()
for f in files:
    i = 0
    with open(f) as fi:
        print(f)
        for line in fi:
            tokens = line.split(",")
            if len(tokens) == 3:
                movies_counter[movie] += 1
                i += 1
                user, rating, date = tokens
                first_date = min(first_date, date)
                last_date = max(first_date, date)
                reduced_date = date[:date.rfind("-")]
                dates_counter[reduced_date] += 1
                users_counter[user] += 1
                if i == 20000:
                    break
            else:
                movie = line[:line.find(":")]


print()
print("first date: ", first_date)
print("last date: ", last_date)

keys = list(dates_counter.keys())
ticks = list(keys)
for i in range(len(ticks)):
    if i % 8 != 0:
        ticks[i] = ""

plt.bar(keys, dates_counter.values(), color='r')
plt.axes().get_xaxis().set_ticklabels(ticks)
plt.show()

# num_buckets = 5
# max_val = -999999
# min_val = 999999
# for k in movies_counter:
#     v = movies_counter[k]
#     max_val = max(max_val, v)
#     min_val = min(min_val, v)
# print(max_val)
# print(min_val)
# range_vals = max_val - min_val + 1
# bucket_size = range_vals / num_buckets
# buckets = Counter()
# for k in movies_counter:
#     v = movies_counter[k]
#     bucket = str(int((v - min_val) / bucket_size))
#     # print(bucket)
#     buckets[bucket] += 1
#
# ticks = list(buckets.keys())
# lo = min_val
# for i in range(len(ticks)):
#     hi = lo + bucket_size
#     ticks[i] = str(lo) + "-" + str(hi)
#     lo = hi
#
# print((buckets.keys()))
# plt.bar(ticks, buckets.values(), color='r')
# plt.show()
max_val = -9999999
min_val = 9999999
ranges = [1000, 2000, 3000, 4000, 5000, 6000]
for k in movies_counter:
    v = movies_counter[k]
    max_val = max(max_val, v)
    min_val = min(min_val, v)
print(max_val)
print(min_val)

buckets = Counter()
for k in movies_counter:
    v = movies_counter[k]
    bucket = ranges[-1]
    for r in ranges:
        if v <= (r + 0.5):
            bucket = r
            break
    buckets[bucket] += 1

print(buckets)
ticks = ["-" + str(k) for k in buckets.keys()]
ticks[-1] = "> " + str(ranges[-1])

print(buckets)
print(ticks)
plt.bar(ticks, buckets.values(), color='r')
plt.show()

max_val = -9999999
min_val = 9999999
ranges = [1, 2, 3, 4, 5, 6, 7]
for k in users_counter:
    v = users_counter[k]
    max_val = max(max_val, v)
    min_val = min(min_val, v)
print(max_val)
print(min_val)

buckets = Counter()
for k in users_counter:
    v = users_counter[k]
    bucket = ranges[-1]
    for r in ranges:
        if v <= (r + 0.5):
            bucket = r
            break
    buckets[bucket] += 1

print(buckets)
ticks = ["-" + str(k) for k in buckets.keys()]
ticks[-1] = "> " + str(ranges[-1])


print(buckets)
print(ticks)
plt.bar(ticks, buckets.values(), color='r')
plt.show()