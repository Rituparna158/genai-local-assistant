x = (1, 2, [3, 4])
try:
    x[2] += [5, 6]
except TypeError as e:
    print(f"Error raised: {e}")

print(f"Resulting tuple: {x}")