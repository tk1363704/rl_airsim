import time

current_time = time.time()

for _ in range(10000):
    j = 1

test_time = time.time()

print(test_time - current_time)

list_ = [-1.0, -1.0, 1, 2, 3]
print(list_.count(-1.0))