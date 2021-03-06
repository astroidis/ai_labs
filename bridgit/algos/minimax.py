from xo import state_xo


def minimax(state, depth, original, player, opponent):
    best_move, best_score = None, None
    moves = state.get_moves(player)
    if (depth == 0) or (moves == []):
        return (None, state.score(player))

    for m in moves:
        state.do_move(m)
        move, score = minimax(state, depth-1, original, opponent, player)
        state.undo_move(m)
        if player == original:
            if (best_score is None) or (score > best_score):
                best_move, best_score = m, score
        else:
            if (best_score is None) or (score < best_score):
                best_move, best_score = m, score

    return (best_move, best_score)


def bestmove(state, depth, player, opponent):
    original = player
    move, score = minimax(state, depth, original, player, opponent)
    return move


if __name__ == "__main__":
    # начальное состояние (пустое)
    s = state_xo()

    # на три хода (на шесть полуходов) вперед
    level = 6

    # первым ходит "X", вторым - "0"
    player, opponent = "X", state_xo.opponent["X"]

    # получаем лучший ход
    move = bestmove(s, level, player, opponent)

    print(f"Best move is: {move}")
