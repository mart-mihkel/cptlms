import logging
import random
from typing import Literal, TypedDict

import numpy as np
from datasets.arrow_dataset import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.trainer_utils import EvalPrediction

logger = logging.getLogger(__name__)


type MultinerdLang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type MultinerdTag = Literal[
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]

MULTINERD_TAG2ID: dict[MultinerdTag, int] = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
}

MULTINERD_ID2TAG: dict[int, MultinerdTag] = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-ANIM",
    8: "I-ANIM",
    9: "B-BIO",
    10: "I-BIO",
    11: "B-CEL",
    12: "I-CEL",
    13: "B-DIS",
    14: "I-DIS",
    15: "B-EVE",
    16: "I-EVE",
    17: "B-FOOD",
    18: "I-FOOD",
    19: "B-INST",
    20: "I-INST",
    21: "B-MEDIA",
    22: "I-MEDIA",
    23: "B-MYTH",
    24: "I-MYTH",
    25: "B-PLANT",
    26: "I-PLANT",
    27: "B-TIME",
    28: "I-TIME",
    29: "B-VEHI",
    30: "I-VEHI",
}


class MultinerdMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


def tokenize_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    data: Dataset,
) -> Dataset:
    def _tokenize(batch: MultinerdBatch) -> BatchEncoding:
        """
        Returns:
            BatchEncoding: with data
                input_ids: list[list[int]]
                attention_mask: list[list[int]]
                labels: list[list[int]]
        """
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        labels = []
        for i, ner_tags in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            tag_ids = _align_words_to_tags(word_ids=word_ids, ner_tags=ner_tags)
            labels.append(tag_ids)

        tokenized["labels"] = labels
        return tokenized

    return data.map(_tokenize, batched=True, remove_columns=data.column_names)


def tokenize_multinerd_prompted(
    tokenizer: PreTrainedTokenizerFast,
    data: Dataset,
    system_prompt_tokens: list[str] = [],
    drop_other_tag_prob: float = 0.98,
) -> Dataset:
    """
    Args:
        drop_other_tag_prob: float, There are disproportionately more other
            type NER tags in the MultiNERD dataset, drop examples with target
            tag other using this probability.
    """
    sep_token = tokenizer.special_tokens_map["sep_token"]
    cls_token = tokenizer.special_tokens_map["cls_token"]

    assert isinstance(cls_token, str), "Expected one classification token"
    assert isinstance(sep_token, str), "Expected one separator token"

    def _tokenize(batch: MultinerdBatch) -> BatchEncoding:
        """
        Returns:
            BatchEncoding: with data
                input_ids: list[list[int]]
                attention_mask: list[list[int]]
                labels: list[int]
        """
        labels = []
        prompts = []
        for tokens, tags in zip(batch["tokens"], batch["ner_tags"]):
            for token, tag in zip(tokens, tags):
                if tag == 0 and random.random() < drop_other_tag_prob:
                    continue

                prompt_tokens = _prepare_prompt_bert(
                    target_token=token,
                    tokens=tokens,
                    system_tokens=system_prompt_tokens,
                    sep_token=sep_token,
                    cls_token=cls_token,
                )

                prompts.append(prompt_tokens)
                labels.append(tag)

        tokenized = tokenizer(
            prompts,
            truncation=True,
            is_split_into_words=True,
        )

        tokenized["labels"] = labels
        return tokenized

    return data.map(
        _tokenize,
        batched=True,
        remove_columns=data.column_names,
    )


def compute_multinerd_prompted_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    prec = precision_score(
        labels,
        preds,
        average="macro",
        zero_division=np.nan,  # type: ignore
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": prec,
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro"),
    }


def _prepare_prompt_bert(
    target_token: str,
    tokens: list[str],
    system_tokens: list[str],
    sep_token: str = "[SEP]",
    cls_token: str = "[CLS]",
) -> list[str]:
    if len(system_tokens) > 0:
        system_tokens.append(sep_token)

    return (
        [cls_token] + system_tokens + [target_token, sep_token] + tokens + [sep_token]
    )


def _align_words_to_tags(
    word_ids: list[int | None],
    ner_tags: list[MultinerdTag],
) -> list[int]:
    label_ids = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            label_ids.append(ner_tags[word_id])
        else:
            label_ids.append(-100)

        previous_word_id = word_id

    return label_ids


def filter_multinerd_english(batch: MultinerdBatch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]
