import heapq

def h(s, goal):                       # heuristic: number of misplaced tiles
    return sum(1 for i in range(9) if s[i] and s[i] != goal[i])

def astar(start, goal):
    open_list = [(h(start, goal), 0, start, [start])]   # (f, g, state, path)
    closed = set()
    while open_list:
        f, g, s, path = heapq.heappop(open_list)       # node with least f
        if s == goal:
            return path
        if s in closed:
            continue
        closed.add(s)
        z = s.index(0)
        r, c = divmod(z, 3)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):   # up, down, left, right
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                n = list(s)
                n[z], n[nr*3 + nc] = n[nr*3 + nc], n[z]
                n = tuple(n)
                if n not in closed:
                    heapq.heappush(open_list, (g + 1 + h(n, goal), g + 1, n, path + [n]))
    return None

def read(name):
    print(f"Enter {name} state (3 rows, use 0 for blank):")
    return tuple(int(x) for _ in range(3) for x in input().split())

start, goal = read("initial"), read("goal")
path = astar(start, goal)
if path:
    for step, s in enumerate(path):
        print(f"\nStep {step}:")
        for i in range(0, 9, 3):
            print(*[x if x else '_' for x in s[i:i+3]])
    print("\nTotal moves:", len(path) - 1)
else:
    print("No solution exists")
