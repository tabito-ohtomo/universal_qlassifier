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


class AccuracyTable():
    expected_to_actual: Dict[Label, Dict[Label, int]]

    def __init__(self):
        self.expected_to_actual = {}

    def add(self, expected_label: Label, actual_label: Label):
        expected_label_dict = self.expected_to_actual.get(expected_label, {})
        expected_label_dict[actual_label] = expected_label_dict.get(actual_label, 0) + 1
        self.expected_to_actual[expected_label] = expected_label_dict