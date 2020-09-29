class Player:
    def __init__(self, edges, mark):
        self.edges = edges
        self.mark = mark

    def print_edges(self):
        print(f"Player {self.mark.upper()}{{")
        for key in self.edges.keys():
            print(f"{key}:  {self.edges[key]}")
        print("}")


def print_board(b):
    for row in b:
        print(" ".join(row))


def get_moves(board, player):
    moves = []

    if player.mark == "x":
        for i in range(1, 10, 2):
            for j in range(1, 10, 2):
                if board[i][j] == " ":
                    moves.append((player, i, j))

        for i in range(2, 10, 2):
            for j in range(2, 10, 2):
                if board[i][j] == " ":
                    moves.append((player, i, j))

    if player.mark == "o":
        for i in range(1, 10, 2):
            for j in range(1, 10, 2):
                if board[i][j] == " ":
                    moves.append((player, i, j))

        for i in range(2, 10, 2):
            for j in range(2, 10, 2):
                if board[i][j] == " ":
                    moves.append((player, i, j))

    return moves


# def do_move(board, move):
#     player, i, j = move
#     board[i][j] = player.mark
#     if (player.mark == "x"):
#         # horizontal
#         if (i % 2 == 0) and (j % 2 == 0):
#             player.edges.append(((i, j-1), (i, j+1)))
#         # vertical
#         elif (i % 2 != 0) and (j % 2 != 0):
#             player.edges.append(((i-1, j), (i+1, j)))
#
#     if (player.mark == "o"):
#         # vertical
#         if (i % 2 == 0) and (j % 2 == 0):
#             player.edges.append(((i-1, j), (i+1, j)))
#         # horizontal
#         elif (i % 2 != 0) and (j % 2 != 0):
#             player.edges.append(((i, j-1), (i, j+1)))


def do_move(board, move):
    player, i, j = move
    board[i][j] = player.mark
    if (player.mark == "x"):
        # horizontal
        if (i % 2 == 0) and (j % 2 == 0):
            # player.edges.append(((i, j-1), (i, j+1)))
            player.edges[(i, j)] = ((i, j-1), (i, j+1))
        # vertical
        elif (i % 2 != 0) and (j % 2 != 0):
            # player.edges.append(((i-1, j), (i+1, j)))
            player.edges[(i, j)] = ((i-1, j), (i+1, j))

    if (player.mark == "o"):
        # vertical
        if (i % 2 == 0) and (j % 2 == 0):
            # player.edges.append(((i-1, j), (i+1, j)))
            player.edges[(i, j)] = ((i-1, j), (i+1, j))
        # horizontal
        elif (i % 2 != 0) and (j % 2 != 0):
            # player.edges.append(((i, j-1), (i, j+1)))
            player.edges[(i, j)] = ((i, j-1), (i, j+1))


def undo_move(board, move):
    player, i, j = move
    board[i][j] = " "
    del player.edges[(i, j)]


board = [
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "],
    ["o", " ", "o", " ", "o", " ", "o", " ", "o", " ", "o"],
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "],
    ["o", " ", "o", " ", "o", " ", "o", " ", "o", " ", "o"],
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "],
    ["o", " ", "o", " ", "o", " ", "o", " ", "o", " ", "o"],
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "],
    ["o", " ", "o", " ", "o", " ", "o", " ", "o", " ", "o"],
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "],
    ["o", " ", "o", " ", "o", " ", "o", " ", "o", " ", "o"],
    [" ", "x", " ", "x", " ", "x", " ", "x", " ", "x", " "]
]

player_red = Player({}, "x")
player_blue = Player({}, "o")

moves = get_moves(board, player_red)
do_move(board, moves[0])

moves = get_moves(board, player_blue)
do_move(board, moves[0])

moves = get_moves(board, player_red)
do_move(board, moves[0])

moves = get_moves(board, player_blue)
do_move(board, moves[0])

moves = get_moves(board, player_red)
do_move(board, moves[0])

moves = get_moves(board, player_blue)
do_move(board, moves[0])

moves = get_moves(board, player_red)
do_move(board, moves[0])

moves = get_moves(board, player_blue)
do_move(board, moves[0])

# do_move(board, moves[0])
# do_move(board, moves[1])
# do_move(board, moves[2])
# do_move(board, moves[3])

print_board(board)
print()
player_blue.print_edges()
print()
player_red.print_edges()


print("\n==================\n")
undo_move(board, moves[0])
print_board(board)
print()
player_blue.print_edges()

# print_board(board)
# print(player_blue.edges)
