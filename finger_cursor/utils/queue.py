import numpy as np

from .registry import Registry


QUEUE = Registry("QUEUE")


class Queue:
    """
    A queue with probably a fixed size.
    Whenever it is initialized, it will be registered to QUEUE as a global variable
    so that any file in this project can access it through queue() method
    """
    def __init__(self, name, capacity):
        self._queue = []
        self.name = name
        self.capacity = capacity
        QUEUE.register(self, name)

    def add(self, x):
        self._queue.append(x)
        if self.capacity > 0 and len(self._queue) > self.capacity:
            self._queue = self._queue[1:]

    def set_capacity(self, capacity):
        if capacity > 0:
            self.capacity = capacity
            if len(self._queue) > self.capacity:
                self._queue = self._queue[-self.capacity:]

    def __getitem__(self, key):
        if isinstance(key, (list, tuple, np.ndarray)):
            return [self._queue[k] for k in key if 0 <= k < len(self._queue)]
        return self._queue[key]

    def __len__(self):
        return len(self._queue)

    def get_all(self):
        return self._queue


def queue(name, capacity=-1):
    """
    If the queue named NAME exists, return it, or create a new Queue object.
    """
    if name in QUEUE:
        target = QUEUE.get(name)
        target.set_capacity(capacity)
        return target

    return Queue(name, capacity)
