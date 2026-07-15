import numpy as np
from collections import defaultdict
from utils import encode, label2id, MAX_LENGTH
import ast
import pandas as pd
import random
from math import ceil
from torch.utils.data import Sampler


def compute_debate_weights(
    debates: list, debate_to_indices: dict, alpha: float
) -> np.ndarray:
    """
    Compute weights for each debate based on their sequence counts and a tempered sampling parameter.
    @param debates: List of debate IDs.
    @param debate_to_indices: Dictionary mapping debate IDs to their sequence indices.
    @param alpha: Tempered sampling parameter.
    @return: Numpy array of debate weights.
    """
    counts = np.array([len(debate_to_indices[d]) for d in debates])
    weights = counts**alpha
    weights = weights / weights.sum()
    return weights


def compute_debate_to_indices(train_data: pd.DataFrame) -> dict:
    """
    Compute a mapping from debate IDs to their sequence indices.
    @param train_data: Training data as a pandas DataFrame.
    @return: Dictionary mapping debate IDs to lists of sequence indices.
    """
    debate_to_indices = defaultdict(list)
    for idx, row in enumerate(train_data.to_dict("records")):
        debate_to_indices[row["debate_id"]].append(idx)
    return debate_to_indices


def compute_debate_to_bio_scores(train_data, tokenizer):
    """
    Compute BIO scores for each debate based on their sequence labels.
    @param train_data: Training data as a Torch Dataset.
    @return: Dictionary mapping debate IDs to lists of BIO scores.
    """
    debate_to_bio_scores = defaultdict(list)
    B_ID = label2id["B"]
    I_ID = label2id["I"]

    records = train_data.to_dict("records")

    for idx, row in enumerate(records):
        enc = encode(
            row["text"],
            ast.literal_eval(row["spans"]),
            tokenizer,
            max_length=MAX_LENGTH,
        )

        labels = np.array(enc["labels"])
        labels = labels[
            np.array(enc["attention_mask"]) == 1
        ]  # Only consider non-padding tokens

        # CRF-relevant BIO signal
        score = 2 * np.sum(labels == B_ID) + 0.5 * np.sum(labels == I_ID)

        # normalization by sequence length to avoid bias towards longer sequences
        score = score / len(labels)

        debate_to_bio_scores[row["debate_id"]].append(float(score))
        del enc, labels
    del records
    return debate_to_bio_scores


class TemperedBatchSampler(Sampler):
    def __init__(
        self,
        debate_to_indices: list[list],
        debate_to_bio_scores: list[list],
        debates: list,
        debate_weights: list,
        batch_size: int,
        num_batches: int,
        bio_alpha: float,
        bio_eps: float,
        debate_alpha: float,
        shuffle: bool = False,
    ):
        self.debate_to_indices = debate_to_indices
        self.debate_to_bio_scores = debate_to_bio_scores
        self.debates = debates
        self.debate_weights = debate_weights
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle

        self.bio_alpha = bio_alpha
        self.bio_eps = bio_eps

        self.use_debate_tempering = debate_alpha != 1.0
        self.use_bio_tempering = bio_alpha != 0

    def _build_normal_debate_schedule(self):
        schedule = []

        for debate in self.debates:
            debate_batches = ceil(len(self.debate_to_indices[debate]) / self.batch_size)
            schedule.extend([debate] * debate_batches)

        if len(schedule) < self.num_batches:
            repeats = ceil(self.num_batches / max(len(schedule), 1))
            schedule = (schedule * repeats)[: self.num_batches]
        else:
            schedule = schedule[: self.num_batches]

        if self.shuffle:
            random.shuffle(schedule)
        return schedule

    def _normal_sample_from_debate(self, debate):
        pool = self.debate_pools[debate]
        ptr = self.debate_ptrs[debate]

        if ptr >= len(pool):
            pool = self.debate_to_indices[debate].copy()

            if self.shuffle:
                random.shuffle(pool)

            self.debate_pools[debate] = pool
            self.debate_ptrs[debate] = 0
            ptr = 0

        batch = pool[ptr : ptr + self.batch_size]
        self.debate_ptrs[debate] += len(batch)

        if len(batch) == 0:
            raise RuntimeError(
                f"Debate '{debate}' produced an empty batch. Check batch_size and debate sampling."
            )

        return batch

    def _sample_from_debate(self, debate):
        """
        Sample sequences from a given debate based on BIO scores.
        @param debate: Debate ID to sample from.
        @return: List of sampled indices for the batch."""
        if not self.use_bio_tempering:
            return self._normal_sample_from_debate(debate)

        indices = self.debate_to_indices[debate]
        bio_scores = self.debate_to_bio_scores[debate]

        weights = np.array(bio_scores, dtype=np.float32)
        weights = np.power(weights + self.bio_eps, self.bio_alpha)
        weights /= weights.sum()

        chosen = np.random.choice(
            indices, size=self.batch_size, replace=False, p=weights
        )

        return chosen.tolist()

    def __iter__(self):
        if self.use_debate_tempering:
            self.debate_schedule = np.random.choice(
                self.debates,
                size=self.num_batches,
                p=self.debate_weights,
            )
        else:
            self.debate_schedule = self._build_normal_debate_schedule()

        if not self.use_bio_tempering:
            self.debate_pools = {}
            self.debate_ptrs = {}

            for d in self.debates:
                pool = self.debate_to_indices[d].copy()

                if self.shuffle:
                    random.shuffle(pool)

                self.debate_pools[d] = pool
                self.debate_ptrs[d] = 0

        for i in range(self.num_batches):
            # Tempered sampling of debates at the batch level
            d = self.debate_schedule[i]

            # Tempered sampling of sequences within the chosen debate based on BIO scores
            batch = self._sample_from_debate(d)

            yield batch

    def __len__(self):
        return self.num_batches
