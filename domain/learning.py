import datetime
from typing import TypeAlias, List, Tuple, Any, Dict

import numpy as np

# Data: TypeAlias = List[Tuple[np.ndarray[Any] | float, int]]
Data: TypeAlias = List[float | int]

Label: TypeAlias = int

LabeledData : TypeAlias = Tuple[Data, Label]
LabeledDataSet: TypeAlias = List[LabeledData]

LabeledDataSetToLearn: TypeAlias = Tuple[LabeledDataSet, LabeledDataSet]


def split_dataset(dataset: LabeledDataSet, number_to_train: int) -> LabeledDataSetToLearn:
    if len(dataset) >= number_to_train:
        raise Exception("invalid number_to_train")
    return dataset[:number_to_train], dataset[number_to_train:]


class LabelCount:
    count: Dict[Label, int]

    def __init__(self):
        self.count = {}

    def on_label(self, label: Label) -> int:
        return self.count.get(label, 0)

    def increment(self, label: Label):
        self.count[label] = self.count.get(label, 0) + 1

    def sum(self) -> int:
        return sum(self.count.values())

    def __str__(self):
        return f"LabelCount(count={self.count})"

    def __repr__(self):
        return f"LabelCount(count={self.count})"


class AccuracyTable():
    expected_to_actual: Dict[Label, LabelCount]

    def __init__(self):
        self.expected_to_actual = {}

    def __str__(self):
        return f"AccuracyTable(expected_to_actual={self.expected_to_actual})"

    def add(self, expected_label: Label, actual_label: Label):
        expected_label_dict: LabelCount = self.expected_to_actual.get(expected_label, LabelCount())
        expected_label_dict.increment(actual_label)
        self.expected_to_actual[expected_label] = expected_label_dict

    def get_accuracy(self) -> float:
        whole_number = sum(map(lambda dict_entry: dict_entry.sum(), self.expected_to_actual.values()))
        accurate_number = sum(map(
            lambda label: self.expected_to_actual.get(label, LabelCount()).on_label(label),
            self.expected_to_actual.keys()))
        print(str(datetime.datetime.now()) + str(self.expected_to_actual))
        return accurate_number / whole_number

class LearningState:
    start_date: datetime.datetime
    end_date: datetime.datetime
    accuracies: List[float] = []

    def __init__(self):
        self.start_date = datetime.datetime.now()

    def add_accuracy(self, accuracy: float):
        self.accuracies.append(accuracy)

    def __str__(self):
        return f"LearningState(start_date={self.start_date}, end_date={self.end_date}, accuracies={self.accuracies})"
