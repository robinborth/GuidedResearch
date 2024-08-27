import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from prettytable import PrettyTable

log = logging.getLogger()


@dataclass
class Track:
    task_name: str
    parent: str | None = None
    start_time: float = 0
    end_time: float = 0
    time_ms: float = 0
    step: int = 0


class TimeTracker:
    def __init__(self) -> None:
        self.tracks: dict[str, list[Track]] = defaultdict(list)
        self.active_task_names: list[str] = []
        self.active_tracks: list[Track] = []

    def start(self, task_name: str, stop: bool = False):
        # quick debugging in a method
        if stop:
            self.stop()

        # compute the current step
        step = 0
        if task_name in self.tracks:
            step = self.tracks[task_name][-1].step + 1

        # find the current parent
        parent = self.active_task_names[-1] if self.active_task_names else None

        # start time
        start_time = time.time()

        # add the track to the queue
        self.active_task_names.append(task_name)
        track = Track(
            task_name=task_name,
            parent=parent,
            start_time=start_time,
            step=step,
        )
        self.active_tracks.append(track)

    def stop(self, task_name: str | None = None):
        end_time = time.time()
        if task_name is not None:  # to ensure correct order
            assert self.active_task_names[-1] == task_name
        self.active_task_names.pop()
        track = self.active_tracks.pop()
        track.end_time = end_time
        track.time_ms = (end_time - track.start_time) * 1000  # in ms
        self.tracks[track.task_name].append(track)

    def compute_statistics(self, round_digits: int = 3):
        out = {}
        for task_name, tracks in self.tracks.items():
            times = np.asarray([float(t.time_ms) for t in tracks])
            out[task_name] = dict(
                task_name=task_name,
                mean=np.mean(times).round(round_digits),
                min=np.min(times).round(round_digits),
                max=np.max(times).round(round_digits),
                median=np.median(times).round(round_digits),
                std=np.std(times).round(round_digits),
                steps=len(times),
                total=np.sum(times).round(round_digits),
            )
        return out

    def parents_mapping(self):
        out = defaultdict(list)
        for task_name, tracks in self.tracks.items():
            parent = tracks[-1].parent
            out[parent].append(task_name)
        return out

    def clean_stats(self, stats):
        _stats = {}
        for k, v in stats.items():
            _stats[k] = "-"
        _stats["mean"] = stats["mean"]
        _stats["task_name"] = stats["task_name"]
        return _stats

    def print_summary(self):
        summary = ""
        statistics = self.compute_statistics()
        for parent, childs in self.parents_mapping().items():
            table = PrettyTable()
            table.field_names = list(statistics[list(statistics.keys())[0]].keys())
            table.align = "r"
            if parent is not None:
                stats = statistics[parent]
                table.add_row(stats.values(), divider=True)
            for task_name in childs:
                stats = statistics[task_name]
                table.add_row(stats.values())
            table_text = "\n" + table.get_string()
            summary += table_text
            log.info(table_text)
        return summary
