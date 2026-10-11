import dataclasses
import heapq
from enum import Enum
from typing import Optional, Iterable


@dataclasses.dataclass(frozen=True)
class Task:
    id: str
    deadline: int
    prerequisites: list[str, ...] = ()

class Scheduler:
    def __init__(self):
        self._heap: list[tuple[int, str]] = []     # (deadline, id)
        self._by_id: dict[str, Task] = {}    # id -> Task

    def add_tasks(self, tasks: Task | Iterable[Task]) -> None:
        if isinstance(tasks, Task):
            tasks = (tasks,)
        for t in tasks:
            if t.id in self._by_id:
                raise ValueError(f"duplicate task id: {t.id}")
            self._by_id[t.id] = t
            heapq.heappush(self._heap, (t.deadline, t.id))

    def consume_task(self) -> Optional[Task]:
        if not self._heap:
            return None
        _, tid = heapq.heappop(self._heap)
        return self._by_id.pop(tid)

    def __len__(self) -> int:
        return len(self._by_id)


class State(Enum):
    WAITING = "WAITING"     # has remaining prereqs
    RUNNABLE = "RUNNABLE"   # in the heap, waiting for a worker
    LEASED = "LEASED"       # handed to a worker, clock running
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"       # retries exhausted
    CANCELLED = "CANCELLED" # upstream task failed permanently s
