def draw_vertical_die(board, i, j):
    i_start, i_stop, i_step = i
    j_start, j_stop, j_step = j

    for i in range(i_start, i_stop, i_step):
        for j in range(j_start, j_stop, j_step):
            if board[i][j] == " ":
                board[i][j] = "*"
                if (i == i_start) or (i == i_stop-1):
                    board[i][(j_stop+j_start)//2] = "*"


def draw_horizontal_die(board, i, j):
    i_start, i_stop, i_step = i
    j_start, j_stop, j_step = j

    for i in range(i_start, i_stop, i_step):
        for j in range(j_start, j_stop, j_step):
            if board[i][j] == " ":
                board[i][j] = "*"
                if (j == j_start) or (j == j_stop-1):
                    board[(i_stop+i_start)//2][j] = "*"


def print_board(board):
    for i in board:
        print(" ".join(i))


def hash_board(board):
    val = 0
    for line in board:
        for cell in line:
            if cell == "*":
                val += 1

    return val


board = [
    [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    [" ", "3", " ", "3", " ", "1", " ", "1", " ", "0", " "],
    [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    [" ", "1", " ", "2", " ", "3", " ", "0", " ", "3", " "],
    [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    [" ", "1", " ", "0", " ", "2", " ", "0", " ", "2", " "],
    [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
    [" ", "1", " ", "2", " ", "3", " ", "3", " ", "0", " "],
    [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
]


print(hash_board(board))

draw_horizontal_die(board, (2, 5, 2), (2, 7, 1))
print_board(board)

print()
print(hash_board(board))
