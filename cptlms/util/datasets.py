from datasets import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerFast


def tokenize_swag(
    swag: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    endings = ["ending0", "ending1", "ending2", "ending3"]

    def _preprocess(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Make four copies of `sent1` field and combine each of them with `sent2`
        to recreate how a sentence starts.

        Combine `sent2` with each of the four possible sentence endings.

        Flatten for tokenization.

        Unflatten afterward so each example has a corresponding `input_ids`,
        `attention_mask`, and `labels` field.
        """
        first_sentences = [[context] * 4 for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][header_idx]}" for end in endings]
            for header_idx, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(
            first_sentences, second_sentences, truncation=True
        )

        return {
            k: [v[i : i + 4] for i in range(0, len(v), 4)]
            for k, v in tokenized_examples.items()
        }

    return swag.map(_preprocess, batched=True)


def tokenize_squad(
    squad: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
) -> DatasetDict:
    def _preprocess(examples: dict) -> BatchEncoding:
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # TODO: refactor
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        return inputs

    return squad.map(
        _preprocess,
        batched=True,
        remove_columns=squad["train"].column_names,
    )
