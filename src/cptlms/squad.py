import logging
from collections import defaultdict
from functools import cached_property
from typing import Annotated, TypedDict, cast

import evaluate
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from transformers.data.data_collator import default_data_collator

logger = logging.getLogger("cptlms")

type OffsetMapping = list[list[tuple[int, int]]]


class SquadMetrics(TypedDict):
    exact_match: float
    f1: float


class SquadRawAnswer(TypedDict):
    answer_start: tuple[int]
    text: tuple[str]


class SquadBatchRaw(TypedDict):
    id: list[int]
    question: list[str]
    context: list[str]
    answers: list[SquadRawAnswer]


class SquadTrainRecord(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    start_positions: int
    end_positions: int


class SquadValRecord(TypedDict):
    input_ids: list[int]
    attention_mask: list[int]
    offset_mpping: list[None | tuple[int, int]]
    example_id: str


class Squad:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 384,
        stride: int = 128,
        train_split: str = "train",
        eval_split: str = "validation",
    ):
        logger.info("init squad")

        train, eval = load_dataset("squad", split=[train_split, eval_split])
        assert isinstance(train, Dataset)
        assert isinstance(eval, Dataset)

        self.train = train
        self.eval = eval

        self.metric = evaluate.loading.load("squad")

        self.max_len = max_len
        self.stride = stride
        self.tokenizer = tokenizer

        self.train_tok, self.eval_tok = self._tokenize()

    def _tokenize(self) -> tuple[Dataset, Dataset]:
        logger.info("tokenize squad")

        train = self.train.map(
            self._preprocess_train_batch,
            batched=True,
            remove_columns=self.train.column_names,
        )

        eval = self.eval.map(
            self._preprocess_eval_batch,
            batched=True,
            remove_columns=self.eval.column_names,
        )

        return train, eval

    def _preprocess_train_batch(self, examples: SquadBatchRaw):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self.stride,
            max_length=self.max_len,
        )

        answers = examples["answers"]
        offset_mapping: OffsetMapping = inputs.pop("offset_mapping")
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

    def _preprocess_eval_batch(self, examples: SquadBatchRaw):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=self.stride,
            max_length=self.max_len,
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

    def compute_metrics(
        self,
        start_logits: Annotated[Tensor, "batch seq"],
        end_logits: Annotated[Tensor, "batch seq"],
    ) -> SquadMetrics:
        predicted_answers = self._postprocess_predictions(
            start_logits=start_logits,
            end_logits=end_logits,
        )

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in self.eval
        ]

        metrics = self.metric.compute(  # type: ignore[missing-argument]
            predictions=predicted_answers,
            references=theoretical_answers,
        )

        return cast(SquadMetrics, metrics)

    def _postprocess_predictions(
        self,
        start_logits: Annotated[Tensor, "batch seq"],
        end_logits: Annotated[Tensor, "batch seq"],
    ) -> list[dict[str, str | int]]:
        predicted_answers = []
        for example in tqdm(self.eval, desc="Postprocess"):
            example_id = example["id"]
            answers = self._extract_answers(
                start_logits=start_logits,
                end_logits=end_logits,
                context=example["context"],
                example_features=self._example_to_features[example_id],
                offset_mapping=self.eval_tok["offset_mapping"],
            )

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        return predicted_answers

    @cached_property
    def _example_to_features(self) -> dict[int, list[int]]:
        example_to_features: dict[int, list[int]] = defaultdict(list)
        for idx, feature in enumerate(self.eval_tok):
            example_to_features[feature["example_id"]].append(idx)

        return example_to_features

    @staticmethod
    def _extract_answers(
        start_logits: Annotated[Tensor, "batch seq"],
        end_logits: Annotated[Tensor, "batch seq"],
        context: list[str],
        example_features: list[int],
        offset_mapping: OffsetMapping,
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

    @staticmethod
    def collate_fn(batch: list[SquadTrainRecord | SquadValRecord]) -> dict[str, Tensor]:
        for item in batch:
            item = cast(dict, item)
            item.pop("token_type_ids", None)
            item.pop("offset_mapping", None)
            item.pop("example_id", None)

        return default_data_collator(batch)
