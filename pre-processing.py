import numpy as np
from sklearn.model_selection import train_test_split

types_dict = {}

PEPTIDS_TO_INDEX = {'C': 0, 'L': 1, 'M': 2, 'V': 3, 'P': 4, 'I': 5, 'Q': 6, 'R': 7, 'F': 8, 'T': 9, 'A': 10, 'H': 11, 'D': 12, 'W': 13, 'N': 14, 'K': 15, 'S': 16, 'G': 17, 'Y': 18, 'E': 19}


def create_peptids_dict():
    peptids_set = set()
    data = open(r"data\neg_A0201.txt", "r")
    for line in data:
        for peptid in line:
            peptids_set.add(peptid)
    peptids_set.remove("\n")
    result = dict(zip(list(peptids_set), range(20)))
    print(result)


def represent_data_as_vector(data, label):
    lines = data.readlines()
    matrix_representation = np.zeros(shape=(len(lines), 9*20))
    for j in range(len(lines)):
        for i in range(9):
            peptid = lines[j][i]
            peptid_idx = PEPTIDS_TO_INDEX[peptid]
            matrix_representation[j][i*20 + peptid_idx] = 1

    return matrix_representation, np.array([label] * len(lines))


def pre_process_labeled_data():
    neg = open(r"data\neg_A0201.txt", "r")
    pos = open(r"data\pos_A0201.txt", "r")

    data_neg, labels_neg = represent_data_as_vector(neg, 0)
    data_pos, labels_pos = represent_data_as_vector(pos, 1)

    all_data = np.concatenate((data_neg, data_pos))
    labels = np.concatenate((labels_neg, labels_pos))

    np.savetxt(fname=r"data\data.txt", X=all_data)
    np.savetxt(fname=r"data\labels.txt", X=labels)

    x_train, x_test, y_train, y_test = train_test_split(all_data, labels, test_size=0.01)

    np.savetxt(fname=r"data\x_train.txt", X=x_train)
    np.savetxt(fname=r"data\x_test.txt", X=x_test)
    np.savetxt(fname=r"data\y_train.txt", X=y_train.reshape(-1,1))
    np.savetxt(fname=r"data\y_test.txt", X=y_test.reshape(-1,1))


def pre_process_protein_spike_data():
    spike_protein_str = open('data\spike_protein_data.txt').read().replace('\n', '')
    data_matrix = np.zeros(shape=(len(spike_protein_str) - 8, 180))
    for i in range(len(spike_protein_str) - 8):
        for j in range(9):
            data_matrix[i, j * 20 + PEPTIDS_TO_INDEX[spike_protein_str[i + j]]] = 1
    np.savetxt('data\spike_protein_matrix.txt', data_matrix)


def main():
    pre_process_labeled_data()
    pre_process_protein_spike_data()

if __name__ == "__main__":
    main()



