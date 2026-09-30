import math

def alphabeta(depth, idx, is_max, values, alpha, beta, h):
    if depth == h:                          # leaf node
        return values[idx]
    best = -math.inf if is_max else math.inf
    for i in range(2):                      # two children per node
        val = alphabeta(depth + 1, idx * 2 + i, not is_max, values, alpha, beta, h)
        if is_max:
            best = max(best, val)
            alpha = max(alpha, best)
        else:
            best = min(best, val)
            beta = min(beta, best)
        if beta <= alpha:                   # prune
            break
    return best

values = list(map(int, input("Enter leaf values: ").split()))
h = int(math.log2(len(values)))
print("Evaluated Value:", alphabeta(0, 0, True, values, -math.inf, math.inf, h))
