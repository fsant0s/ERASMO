import random
import typing as tp
import inflect
import re

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class ErasmoDataset(Dataset):
    """ Erasmo Dataset

    The ErasmoDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """
    has_printed = False
    def set_text_to_num(self, text_to_num: bool):
        self.text_to_num = text_to_num
        if self.text_to_num:
            self.p = inflect.engine()

    def set_tokenizer(self, tokenizer):
        """ Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)
    
        shuffled_text = ", ".join(
            ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
        )

        if self.text_to_num:
            shuffled_text =  re.compile(r'\d+(\.\d+)?').sub(
                lambda match: f"{self.p.number_to_words(match.group(0).replace('.', ' point '))}" if '.' in match.group(0) else self.p.number_to_words(match.group(0)),
                shuffled_text
            )
        
        if not ErasmoDataset.has_printed:
            print("Random sample:", shuffled_text)
            ErasmoDataset.has_printed = True

        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text
        
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)
        

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)
    
@dataclass
class ErasmoDataCollator(DataCollatorWithPadding):
    """ Erasmo Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
