from typing import TypeAlias, List, Tuple, Any

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


