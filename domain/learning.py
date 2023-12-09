from typing import TypeAlias, List, Tuple, Any

import numpy as np

Data: TypeAlias = List[Tuple[np.ndarray[Any] | float, int]]

Label: TypeAlias = int

LabeledDataSet: TypeAlias = List[Tuple[Data, Label]]

LabeledDataSetToLearn: TypeAlias = Tuple[LabeledDataSet, LabeledDataSet]


def split_dataset(dataset: LabeledDataSet, number_to_train: int) -> LabeledDataSetToLearn:
    if len(dataset) >= number_to_train:
        raise Exception("invalid number_to_train")
    return dataset[:number_to_train], dataset[number_to_train:]


