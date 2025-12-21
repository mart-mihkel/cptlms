import logging
from collections import defaultdict
from typing import Annotated, TypedDict, cast

import evaluate
from datasets.arrow_dataset import Dataset
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)
squad_metric = evaluate.loading.load("squad")


class SquadMetrics(TypedDict):
    exact_match: float
    f1: float


class SquadAnswer(TypedDict):
    answer_start: tuple[int]
    text: tuple[str]


class SquadBatch(TypedDict):
    id: list[int]
    question: list[str]
    context: list[str]
    answers: list[SquadAnswer]


class SquadTrainBatch(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    start_positions: int
    end_positions: int


class SquadValBatch(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    offset_mpping: list[None | tuple[int, int]]
    example_id: str


def tokenize_squad(
    tokenizer: PreTrainedTokenizerFast,
    data: Dataset,
    max_len: int = 384,
) -> Dataset:
    def _tokenize(batch: SquadBatch) -> BatchEncoding:
        questions = [q.strip() for q in batch["question"]]
        inputs = tokenizer(
            questions,
            batch["context"],
            padding="max_length",
            truncation="only_second",
            max_length=max_len,
            return_offsets_mapping=True,
        )

        offset_mapping: list[list[tuple[int, int]]] = inputs.pop("offset_mapping")
        answers: list[SquadAnswer] = batch["answers"]
        start_positions: list[int] = []
        end_positions: list[int] = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            start, end = _find_label_span(offset, sequence_ids, start_char, end_char)
            start_positions.append(start)
            end_positions.append(end)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    return data.map(
        _tokenize,
        batched=True,
        remove_columns=data.column_names,
    )


def compute_metrics(
    examples: Dataset,
    examples_tokenized: Dataset,
    start_logits: Annotated[Tensor, "batch seq"],
    end_logits: Annotated[Tensor, "batch seq"],
) -> SquadMetrics:
    predicted_answers = _postprocess_predictions(
        examples=examples,
        examples_tokenized=examples_tokenized,
        start_logits=start_logits,
        end_logits=end_logits,
    )

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]

    metrics = squad_metric.compute(  # type: ignore
        predictions=predicted_answers,
        references=theoretical_answers,
    )

    return cast(SquadMetrics, metrics)


def _postprocess_predictions(
    examples: Dataset,
    examples_tokenized: Dataset,
    start_logits: Annotated[Tensor, "batch seq"],
    end_logits: Annotated[Tensor, "batch seq"],
) -> list[dict[str, str | int]]:
    example_to_features: dict[int, list[int]] = defaultdict(list)
    for idx, feature in enumerate(examples_tokenized):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples, desc="Postprocess"):
        example_id = example["id"]
        answers = _extract_answers(
            start_logits=start_logits,
            end_logits=end_logits,
            context=example["context"],
            example_features=example_to_features[example_id],
            offset_mapping=examples_tokenized["offset_mapping"],
        )

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    return predicted_answers


def _find_label_span(
    offset: list[tuple[int, int]],
    sequence_ids: list[int | None],
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
        return 0, 0

    idx = context_start
    while idx <= context_end and offset[idx][0] <= start_char:
        idx += 1

    start = idx - 1

    idx = context_end
    while idx >= context_start and offset[idx][1] >= end_char:
        idx -= 1

    end = idx + 1

    return start, end


def _extract_answers(
    start_logits: Annotated[Tensor, "batch seq"],
    end_logits: Annotated[Tensor, "batch seq"],
    context: list[str],
    example_features: list[int],
    offset_mapping: list[list[tuple[int, int]]],
    n_best: int = 20,
    max_answer_len: int = 30,
) -> list[dict[str, str | int]]:
    answers = []
    for f_idx in example_features:
        start_logit = start_logits[f_idx]
        end_logit = end_logits[f_idx]
        offsets = offset_mapping[f_idx]

        start_indexes = start_logit.argsort(descending=True)[:n_best]
        end_indexes = end_logit.argsort(descending=True)[:n_best]
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue

                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_len
                ):
                    continue

                text = context[offsets[start_index][0] : offsets[end_index][1]]
                score = start_logit[start_index] + end_logit[end_index]
                answers.append({"text": text, "logit_score": score})

    return answers


def squad_collate_fn(batch: list[SquadTrainBatch | SquadValBatch]) -> dict[str, Tensor]:
    for item in batch:
        item = cast(dict, item)
        item.pop("token_type_ids", None)
        item.pop("offset_mapping", None)
        item.pop("example_id", None)

    return default_data_collator(batch)
