board = [' '] * 9
win = [(0,1,2),(3,4,5),(6,7,8),
       (0,3,6),(1,4,7),(2,5,8),
       (0,4,8),(2,4,6)]

def show():
    print('\n', board[0], '|', board[1], '|', board[2])
    print('---+---+---')
    print('', board[3], '|', board[4], '|', board[5])
    print('---+---+---')
    print('', board[6], '|', board[7], '|', board[8])

def winner(p):
    return any(all(board[i] == p for i in w) for w in win)

def dfs(p):
    if winner('X'): return 1
    if winner('O'): return -1
    if ' ' not in board: return 0

    scores = []
    for i in range(9):
        if board[i] == ' ':
            board[i] = p
            scores.append(dfs('O' if p == 'X' else 'X'))
            board[i] = ' '
    return max(scores) if p == 'X' else min(scores)

def computer():
    best, move = -2, None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'X'
            score = dfs('O')
            board[i] = ' '
            if score > best:
                best, move = score, i
    board[move] = 'X'

while True:
    show()
    pos = int(input("Enter position (1-9): ")) - 1
    if pos not in range(9) or board[pos] != ' ':
        print("Invalid move")
        continue

    board[pos] = 'O'
    if winner('O'):
        show(); print("Player O wins!"); break
    if ' ' not in board:
        show(); print("DRAW"); break

    computer()
    if winner('X'):
        show(); print("Player X wins!"); break
