import glob
import gzip
import json
import os
import random
import re

import tqdm
import typer
from loguru import logger

app = typer.Typer()


def read_raws(path: str):
    logger.info("Creating features from dataset at {}", path)
    if os.path.isdir(path):
        text_files = glob.glob(f"{path}/*text*.json")
        layout_files = [re.sub("text|txt", "layout", x, 1) for x in text_files]
    else:
        text_files = [path]
        layout_files = [re.sub("text|txt", "layout", path, 1)]
    for text_file, layout_file in zip(text_files, layout_files):
        with open(text_file) as text_reader, open(layout_file) as layout_reader:
            logger.info(f"Start loading {text_file}")
            for i, (text_line, layout_line) in enumerate(
                zip(text_reader, layout_reader)
            ):
                yield json.loads(text_line), json.loads(layout_line)


@app.command()
def create_dataset_spans(
    path: str = typer.Argument(
        ...,
        help="Path to the dataset, like `./train/`",
    ),
    output_file: str = typer.Argument(
        ..., help="Path to the output file, like `./train.jsonl.gz`"
    ),
    src_shuffle_rate: float = typer.Option(
        0.5, help="The rate to shuffle input's order"
    ),
):
    random.seed(42)
    logger.info("Saving features into file {}", output_file)
    f_out = gzip.open(output_file, "wt")
    for text, layout in tqdm.tqdm(read_raws(path)):
        target_boxes = []
        target_texts = []
        last_box = [0, 0, 0, 0]
        last_text = ""
        for s, box in zip(text["tgt"].split(), layout["tgt"]):
            if (not box[2] >= box[0]) or (not box[3] >= box[1]):
                # skip invalid box
                continue
            if (
                box[1] == last_box[1]
                and box[3] == last_box[3]
                and box[0] >= last_box[2]
                and (box[0] - last_box[2])
                < ((last_box[2] - last_box[0]) / max(len(last_text), 1))
            ):
                # merge box of the same line
                last_box[2] = box[2]
                last_text += " " + s
            else:
                if last_text != "":
                    target_boxes.append(last_box.copy())
                    target_texts.append(last_text)
                # renew buffer
                last_box = box.copy()
                last_text = s
        if last_text != "":
            target_boxes.append(last_box.copy())
            target_texts.append(last_text)

        for left, top, right, bottom in target_boxes:
            assert left <= right <= 1000 and top <= bottom <= 1000

        # build source from target
        tmp = [(i, d) for i, d in enumerate(target_boxes)]
        if random.random() < src_shuffle_rate:
            random.shuffle(tmp)
        else:
            # sort from left to right, top to bottom
            tmp = sorted(tmp, key=lambda x: (x[1][2], x[1][1]))

        target_index = [0] * len(target_boxes)
        source_boxes = []
        source_texts = []
        j = 1
        for i, _ in tmp:
            source_boxes.append(target_boxes[i].copy())
            source_texts.append(target_texts[i])
            target_index[i] = j
            j += 1

        f_out.write(
            json.dumps(
                {
                    "source_boxes": source_boxes,
                    "source_texts": source_texts,
                    "target_boxes": target_boxes,
                    "target_texts": target_texts,
                    "target_index": target_index,
                    "bleu": text["bleu"],
                }
            )
            + "\n"
        )
    f_out.close()


if __name__ == "__main__":
    app()
