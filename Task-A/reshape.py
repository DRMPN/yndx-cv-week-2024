import torch


def transpose_from_scratch(lst):
    try:
        len(lst)
    except TypeError:
        return lst
    try:
        len(lst[0])
    except (TypeError, IndexError):
        return lst

    trans_matrix = [[0 for j in range(len(lst))] for i in range(len(lst[0]))]

    for i in range(len(lst)):
        for j in range(len(lst[0])):
            trans_matrix[j][i] = lst[i][j]

    return torch.tensor(trans_matrix)
