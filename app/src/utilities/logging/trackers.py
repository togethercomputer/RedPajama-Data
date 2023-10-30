__all__ = ["RateTracker"]


class RateTracker:
    def __init__(self, n=200):
        self._start_time_tracker = []
        self._counts_tracker = []
        self._n = n

    def update(self, count, start_time):
        if len(self._start_time_tracker) >= self._n:
            self._start_time_tracker.pop(0)
            self._counts_tracker.pop(0)

        self._start_time_tracker.append(start_time)
        self._counts_tracker.append(count)

    def get_rate(self, current_time: float):
        if len(self._start_time_tracker) == 0:
            return 0

        if current_time - self._start_time_tracker[0] < 1e-6:
            return 0

        start_time = self._start_time_tracker[0]
        pages = sum(self._counts_tracker)
        return pages / (current_time - start_time)

    def reset(self):
        self._start_time_tracker = []
        self._counts_tracker = []
