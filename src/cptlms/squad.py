import logging
from collections import defaultdict
from typing import TypedDict, cast

import evaluate
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

type OffsetMapping = list[list[tuple[int, int]]]


class SQuADMetrics(TypedDict):
    exact_match: float
    f1: float


class SQuADBatchAnswer(TypedDict):
    answer_start: tuple[int]
    text: tuple[str]


class SQuADBatch(TypedDict):
    id: list[int]
    question: list[str]
    context: list[str]
    answers: list[SQuADBatchAnswer]


class SQuAD:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_len=384,
        stride=128,
    ):
        logger.info("init SQuAD")

        data = load_dataset("squad")
        assert isinstance(data, DatasetDict)

        self.metric = evaluate.loading.load("squad")

        self.train = data["train"]
        self.val = data["validation"]

        self.max_len = max_len
        self.stride = stride
        self.tokenizer = tokenizer

        self.train_tok, self.val_tok = self._tokenize()

    def _tokenize(self) -> tuple[Dataset, Dataset]:
        logger.info("tokenize SQuAD")

        train = self.train.map(
            self._preprocess_train_batch,
            batched=True,
            remove_columns=self.train.column_names,
        )

        val = self.val.map(
            self._preprocess_val_batch,
            batched=True,
            remove_columns=self.val.column_names,
        )

        return train, val

    def _preprocess_train_batch(self, examples: SQuADBatch):
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

    def _preprocess_val_batch(self, examples: SQuADBatch):
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
        start_logits: NDArray[np.float64],
        end_logits: NDArray[np.float64],
    ) -> SQuADMetrics:
        predicted_answers = self._postprocess_predictions(
            start_logits=start_logits,
            end_logits=end_logits,
        )

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in self.val
        ]

        metrics = self.metric.compute(
            predictions=predicted_answers,
            references=theoretical_answers,
        )

        return cast(SQuADMetrics, metrics)

    def _postprocess_predictions(
        self,
        start_logits: NDArray[np.float64],
        end_logits: NDArray[np.float64],
    ) -> list[dict[str, str | int]]:
        example_to_features: dict[int, list[int]] = defaultdict(list)
        for idx, feature in enumerate(self.val_tok):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(self.val, desc="postprocess"):
            example_id = example["id"]
            answers = self._extract_answers(
                start_logits=start_logits,
                end_logits=end_logits,
                context=example["context"],
                example_features=example_to_features[example_id],
                offset_mapping=self.val_tok["offset_mapping"],
            )

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        return predicted_answers

    @staticmethod
    def _extract_answers(
        start_logits: NDArray[np.float64],
        end_logits: NDArray[np.float64],
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

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
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
