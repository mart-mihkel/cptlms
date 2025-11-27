import logging
from typing import TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class SQuADBatchAnswer(TypedDict):
    answer_start: tuple[int]
    text: tuple[str]


class SQuADBatch(TypedDict):
    id: list[int]
    question: list[str]
    context: list[str]
    answers: list[SQuADBatchAnswer]


class SQuAD:
    _max_len = 384
    _stride = 128

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        logger.info("init SQuAD")

        data = load_dataset("squad")
        assert isinstance(data, DatasetDict)

        self.data = data
        self.tokenizer = tokenizer
        self.tokenized = self._tokenize()

    def _tokenize(self) -> DatasetDict:
        logger.info("tokenize SQuAD")

        train = self.data["train"].map(
            self._preprocess_train_batch,
            batched=True,
            remove_columns=self.data["train"].column_names,
        )

        val = self.data["validation"].map(
            self._preprocess_val_batch,
            batched=True,
            remove_columns=self.data["validation"].column_names,
        )

        return DatasetDict({"train": train, "validation": val})

    def _preprocess_train_batch(self, examples: SQuADBatch):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            stride=self._stride,
            padding="max_length",
            max_length=self._max_len,
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        answers = examples["answers"]
        offset_mapping: list[list[tuple[int, int]]] = inputs.pop("offset_mapping")
        sample_map: list[int] = inputs.pop("overflow_to_sample_mapping")

        start_positions: list[int] = []
        end_positions: list[int] = []
        for i, (offset, sample_idx) in enumerate(zip(offset_mapping, sample_map)):
            answer = answers[sample_idx]
            start_chr = answer["answer_start"][0]
            end_chr = answer["answer_start"][0] + len(answer["text"][0])
            start_pos, end_pos = self._find_label_span(
                offset=offset,
                seq_ids=inputs.sequence_ids(i),
                answer_span=(start_chr, end_chr),
            )

            start_positions.append(start_pos)
            end_positions.append(end_pos)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    def _preprocess_val_batch(self, examples: SQuADBatch):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            stride=self._stride,
            padding="max_length",
            max_length=self._max_len,
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        sample_map: list[int] = inputs.pop("overflow_to_sample_mapping")
        example_ids: list[int] = []
        for i in range(len(inputs.data["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            inputs.data["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(inputs.data["offset_mapping"][i])
            ]

        inputs["example_id"] = example_ids

        return inputs

    @staticmethod
    def _find_label_span(
        offset: list[tuple[int, int]],
        seq_ids: list[int | None],
        answer_span: tuple[int, int],
    ) -> tuple[int, int]:
        start_chr, end_chr = answer_span

        idx = 0
        while seq_ids[idx] != 1:
            idx += 1

        ctx_start = idx

        while seq_ids[idx] == 1:
            idx += 1

        ctx_end = idx - 1

        if offset[ctx_start][0] > end_chr or offset[ctx_end][1] < start_chr:
            return 0, 0

        idx = ctx_start
        while idx <= ctx_end and offset[idx][0] <= start_chr:
            idx += 1

        start_pos = idx - 1

        idx = ctx_end
        while idx >= ctx_start and offset[idx][1] >= end_chr:
            idx -= 1

        end_pos = idx + 1

        return start_pos, end_pos
