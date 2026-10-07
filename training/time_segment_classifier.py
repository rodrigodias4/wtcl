import ast
from datetime import datetime
import json
from pathlib import Path
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from segment_train import SegmentClassifier, SegmentDataset
from train import MAX_LENGTH, encode, get_device, get_tokenizer
from utils import console, progress


def parse_args():
    parser = ArgumentParser(
        description="Measure inference time for a binary segment classifier."
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset CSV."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/roberta-base",
        help="Hugging Face transformer checkpoint to use.",
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        help="Optional JSON file; only the dropout value is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    df = read_csv(args.dataset_path)
    hparams = {}
    if args.hparams_path:
        with open(args.hparams_path, "r") as file:
            hparams = json.load(file)

    device = get_device()
    tokenizer = get_tokenizer(args.model_name)
    dataset = SegmentDataset(df.to_dict("records"), tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if len(dataloader) == 0:
        raise ValueError("The dataset contains no rows to time.")

    model = SegmentClassifier(
        args.model_name,
        dropout=hparams.get("dropout", 0.1),
    ).to(device)
    model.eval()

    task = progress.add_task("Inference", total=len(dataloader))
    progress.start()
    try:
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                if i == 10:
                    break

            torch.cuda.synchronize(device)
            times = []
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                start = time.perf_counter()
                model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize(device)
                end = time.perf_counter()
                times.append(end - start)
                progress.update(task, advance=1)
    finally:
        progress.stop()
        progress.remove_task(task)

    average_time = sum(times) / len(times)
    console.print(
        f"Average segment inference time per batch: {average_time * 1000:.2f} ms"
    )

    with (
        Path(__file__)
        / "times"
        / f"inference_times_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    ).open("w") as file:
        json.dump(times, file)


if __name__ == "__main__":
    main()
