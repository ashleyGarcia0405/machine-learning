# Part 1: Earliest Deadline Task Without Dependencies
# tasks = [(task_id, deadline), ...]
# def solution(tasks: list[tuple[int, int]]) -> int or None:
#     if not tasks:
#         return None
#     return min(tasks, key=lambda t: (t[1], t[0]))[0]

# Part 2: Schedule Ready Tasks by Deadline
# tasks = [(task_id, deadline, prerequisites), ...]
# def solution(tasks: list[tuple[int, int, list[int]]]) -> list[int]:
#     if not tasks:
#         return []
#
#     deadline = {}       # task_id -> deadline
#     dependents = {}     # task_id -> who is waiting on this task
#     remaining = {}      # task_id -> how many prereqs are unfinished
#
#     for task_id, d, prereqs in tasks:
#         deadline[task_id] = d
#         dependents[task_id] = []
#
#     for task_id, d, prereqs in tasks:
#         unique = set(prereqs)
#         remaining[task_id] = len(unique)
#         for p in unique:
#             dependents[p].append(task_id)
#
#     ready = [tid for tid in remaining if remaining[tid] == 0]
#     order = []
#
#     while ready:
#         best = min(ready, key=lambda tid: (deadline[tid], tid))
#         ready.remove(best)
#         order.append(best)
#
#         for child in dependents[best]:
#             remaining[child] -= 1
#             if remaining[child] == 0:
#                 ready.append(child)
#     return order


# Part 3: Deadline-Aware Task Scheduler (OPTIMIZED)
# tasks = [(task_id, deadline, prerequisites), ...]
import heapq

def solution(tasks: list[tuple[int, int, list[int]]]) -> list[int] | ValueError:
    if not tasks:
        return []

    deadline = {}       # task_id -> deadline
    dependents = {}     # task_id -> who is waiting on this task
    remaining = {}      # task_id -> how many prereqs are unfinished

    for task_id, d, prereqs in tasks:
        deadline[task_id] = d
        dependents[task_id] = []

    for task_id, d, prereqs in tasks:
        unique = set(prereqs)
        remaining[task_id] = len(unique)
        for p in unique:
            dependents[p].append(task_id)

    heap = [(deadline[tid], tid) for tid in remaining if remaining[tid] == 0]
    heapq.heapify(heap)
    order = []

    while heap:
        _, task_id = heapq.heappop(heap)
        order.append(task_id)

        for child in dependents[task_id]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(heap, (deadline[child], child))

    # cycle detection
    if len(order) != len(tasks):
        raise ValueError("Cycle detected")

    return order

# main
tasks = [(1, 4, []), (2, 2, []), (3, 3, [1, 2]), (4, 1, [2])]
print(solution(tasks))