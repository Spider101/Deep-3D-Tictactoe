import numpy as np

def empty_state():
    '''establish the empty state wherein each cell is filled by zero'''
    return np.array([[0] * 3] * 3)

def print_board(state):
    board_format = "-------------\n| {0} | {1} | {2} |\n|-------------|\n| {3} | {4} | {5} |\n|-------------|\n| {6} | {7} | {8} |\n-------------"
    cell_values = []
    symbols = ["O", " ", "X"]

    for i in range(len(state)):
        for j in range(len(state[i])):
            cell_values.append(symbols[int(state[i][j] + 1)])

    print(board_format.format(*cell_values))

def open_spots(state):
    open_cells = []
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                open_cells.append(i * len(state) + j)
    return open_cells


def is_game_over(state):
    ''' check if any of the columns or rows or diagonals when summed are
    divisible by the width or height of the board - indication that the
    game has been won. The sign of the sum tells us who has won'''

    for i in range(len(state)):

        state_trans = np.array(state).transpose()  # transposed board state

        # check for winner row-wise
        if np.array(state[i]).sum() != 0 and np.array(state[i]).sum() % 3 == 0:
            return np.array(state[i]).sum() / 3

            # check for winner column-wise
        elif state_trans[i].sum() != 0 and state_trans[i].sum() % 3 == 0:
            return state_trans.sum() / 3

            # extract major diagonal from the state
    major_diag = np.multiply(np.array(state), np.identity(len(state)))
    if major_diag.sum() != 0 and major_diag.sum() % 3 == 0:
        return major_diag.sum() / 3

        # extract minor diagonal from the state
    minor_diag = np.multiply(np.array(state), np.fliplr(major_diag))
    if minor_diag.sum() != 0 and minor_diag.sum() % 3 == 0:
        return minor_diag.sum() / 3

    return 0  # no clear winner


def get_state_key(state):
    ''' convert the state into a unique key by concatenating
    all the flattened values in the state'''

    flat_state = [cell for row in state for cell in row]
    key = "".join(map(str, flat_state))
    #print(key)
    return key
