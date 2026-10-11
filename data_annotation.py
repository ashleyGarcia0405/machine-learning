from collections import defaultdict
from typing import Iterable

# You are building the service that decides which labeling task an annotator picks up next.
# Tasks arrive continuously from customer pipelines. Annotators poll the service when they are free.

class TaskRouter:
    def __init__(self):
        # task dict {id, priority, deadline}
        self.tasks= {}      # task_id -> the task dict
        self.available = set()
        self.skills = {}
        self.refused = defaultdict(set)

        self.remaining = {}     # task_id -> count of unfinished prereqs
        self.dependents = defaultdict(set) # task_id -> set(ids waiting on it)
        self.completed = set()

    def add_tasks(self, tasks) -> None:
        if isinstance(tasks, dict):
            tasks = [tasks]
        for t in tasks:
            self.tasks[t["id"]] = t
            self.available.add(t["id"])

            prereqs = set(t.get("prerequisites", []))
            pending = [p for p in prereqs if not p in self.completed]
            self.remaining[t] = len(pending)
            for p in pending:
                self.dependents[p].add(t["id"])
            if not pending:
                self.available.add(t["id"])

    def next_task(self, annotator_id, now):
        # if not self.available:
        #     return None
        eligible = [tid for tid in self.available if self._offerable(tid, annotator_id)]
        if not eligible:
            return None
        best = min(eligible, key=lambda t: self._sort_key(t, now))
        self.available.remove(best)
        return best

    def _sort_key(self, task_id, now):
        t = self.tasks[task_id]
        if t["deadline"] < now:
            return (0, t["deadline"], 0, task_id)  # overdue
        return (1, -t["priority"], t["deadline"], task_id)  # on time

    def register_annotator(self, annotator_id, skills):
        self.skills[annotator_id] = set(skills)

    def _offerable(self, task_id, annotator_id):
        if annotator_id in self.refused[task_id]:
            return False
        need = self.tasks[task_id].get("required_skill")
        if need is None:
            return True
        return need in self.skills.get(annotator_id, set())

    def skip_task(self, task_id, annotator_id):
        self.refused[task_id].add(annotator_id)

    def complete_task(self, task_id, annotator_id):
        if task_id not in self.completed:
            return
        self.completed.add(task_id)

        for child in self.dependents[task_id].pop(task_id, []):
            self.remaining[child] -= 1
            if self.remaining[child] == 0:
                self.available.add(child)