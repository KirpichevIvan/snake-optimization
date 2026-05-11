from game import SnakeGame, Move

game = SnakeGame(field_size=(5, 5))

while True:
    state = game.get_state()
    field = state.field_matrix()
    for row in field:
        row_str = ""
        for cell in row:
            if cell == 0:
                row_str += " "
            elif cell == 1:
                row_str += "*"
            elif cell == 2:
                row_str += "#"
            elif cell == 3:
                row_str += "$"
        print(row_str)

    print(f"Score: {state.score}")
    print(f"Status: {state.status}")

    move_str = input()
    if move_str == "w":
        move = Move.UP
    elif move_str == "a":
        move = Move.LEFT
    elif move_str == "s":
        move = Move.DOWN
    elif move_str == "d":
        move = Move.RIGHT

    game.step(move)
