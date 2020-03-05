import csv
import random
from numpy import concatenate

transform = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1]
}


def transform_to_ints(gene):
    return [transform.get(x) for x in gene]


def createPairs(data):
    return list(zip(data[::2], data[1::2]))


def createQuartets(data):
    return list(zip(data[::4], data[1::4], data[2::4], data[3::4]))


def flatten(data):
    return list(sum(data, ()))  # prej nejaky hodne drsny rychly solution, z stackoverflow


def concat(data):
    return list(concatenate(data))


i = 0
with open('random_seqs.csv', newline='') as csvfile:
    good_data = open('good_data.csv', 'w', newline='')
    bad_data_shuffle = open('bad_data_shuffle.csv', 'w', newline='')
    bad_data_shuffle_pair = open('bad_data_shuffle_pair.csv', 'w', newline='')
    bad_data_shuffle_quartet = open('bad_data_shuffle_quartet.csv', 'w', newline='')

    good_data_writer = csv.writer(good_data)
    bad_data_shuffle_writer = csv.writer(bad_data_shuffle)
    bad_data_shuffle_pair_writer = csv.writer(bad_data_shuffle_pair)
    bad_data_shuffle_quartet_writer = csv.writer(bad_data_shuffle_quartet)

    for row in csv.reader(csvfile, delimiter=',', quotechar='|'):
        if i == 0:
            i += 1
            continue

        data = row[3]
        if 'N' in data:
            continue

        data = transform_to_ints(data)
        pairs = createPairs(data)
        quartets = createQuartets(data)

        good_data_writer.writerow(concat(data))

        random.shuffle(data)
        random.shuffle(pairs)
        random.shuffle(quartets)

        bad_data_shuffle_writer.writerow(concat(data))
        bad_data_shuffle_pair_writer.writerow(concat(flatten(pairs)))
        bad_data_shuffle_quartet_writer.writerow(concat(flatten(quartets)))

        i += 1
        if i % 2500 == 0:
            print(str(i) + " done.")

    good_data.close()
    bad_data_shuffle.close()
    bad_data_shuffle_pair.close()
    bad_data_shuffle_quartet.close()
print(i)
