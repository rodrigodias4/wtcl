from datetime import datetime
from pathlib import Path
import time
import json
from pandas import read_csv
from train import MAX_LENGTH, WTCLDataset, build_model, get_device, get_tokenizer
from argparse import ArgumentParser
import torch
from utils import console, progress
from torch.utils.data import DataLoader


def parse_args():
    parser = ArgumentParser(
        description="Measure inference time of a trained model on a given dataset."
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset for inference.",
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        required=True,
        help="Path to the hyperparameters file.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for inference."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.dataset_path, "r") as f:
        df = read_csv(f)

    # Load the hyperparameters
    with open(args.hparams_path, "r") as f:
        hparams = json.load(f)

    hparams["batch_size"] = args.batch_size  # Override batch size if provided

    device = get_device()

    # Load the trained model
    model = build_model(
        "FacebookAI/roberta-base",
        hparams=hparams,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path))
    tokenizer = get_tokenizer("FacebookAI/roberta-base")

    dataset = WTCLDataset(df.to_dict("records"), tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )
    times = []

    progress.start()
    task = progress.add_task(description="Warmup", total=10)
    model.eval()
    # warmup
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            crf_mask = batch["crf_mask"].to(device)

            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                crf_mask=crf_mask,
            )
            progress.advance(task)
            if i == 9:
                break  # Only run ten batches for warmup

    progress.remove_task(task)
    task = progress.add_task(description="Inference", total=len(dataloader))
    with torch.inference_mode():
        for batch in dataloader:
            batch_size = batch["input_ids"].size(0)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            crf_mask = batch["crf_mask"].to(device)

            start = time.perf_counter()
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                crf_mask=crf_mask,
            )
            torch.cuda.synchronize()  # Wait for all CUDA operations to finish
            end = time.perf_counter()
            times.append(end - start)

            progress.update(task, advance=1)

    progress.stop()
    progress.remove_task(task)

    with (
        Path(__file__)
        / "times"
        / f"inference_times_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    ).open("w") as f:
        json.dump(times, f)

    avg_time = sum(times) / len(times)
    console.print(f"Average inference time per batch: {avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
