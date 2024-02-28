import glob
import gzip
import json
import os
import pickle
import random
import re

import numpy as np
import tqdm
import typer
from loguru import logger

app = typer.Typer()


def draw_one(ids: list[list], texts: list[str]):
    import cv2

    img = np.empty((1000, 1000, 3), dtype="uint8")
    img.fill(255)
    for i, (_, left, top, right, bottom) in enumerate(ids):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(
            img,
            str(i),
            (left, top - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            img,
            texts[i],
            (left, bottom - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )
    cv2.imshow("img", img)
    return cv2.waitKey(0)


@app.command()
def draw(input_file: str):
    with open(input_file, "rb") as f:
        datas = pickle.load(f)

    for data in datas:
        if draw_one(data["target_ids"], data["target_texts"]) == ord("q"):
            break


def read_raws(path: str) -> list:
    logger.info("Creating features from dataset at {}", path)
    examples = []
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
                if (i + 1) % 10000 == 0:
                    logger.info(f"{i + 1} lines ...")
                examples.append((json.loads(text_line), json.loads(layout_line)))
    return examples


@app.command()
def cache_dataset_spans(
    path: str,
    output_file: str,
    shuffle: bool = True,
    src_shuffle_rate: float = 0.5,
):
    random.seed(42)
    examples = read_raws(path)

    features = []
    for text, layout in tqdm.tqdm(examples):
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

        # if draw_one(source_boxes, source_texts) == ord("q"):
        #     return
        features.append(
            {
                "source_boxes": source_boxes,
                "source_texts": source_texts,
                "target_boxes": target_boxes,
                "target_texts": target_texts,
                "target_index": target_index,
                "bleu": text["bleu"],
            }
        )

    if shuffle:
        random.shuffle(features)

    logger.info("Saving features into cached file {}", output_file)
    with gzip.open(output_file, "wt") as f:
        for feature in tqdm.tqdm(features):
            f.write(json.dumps(feature) + "\n")


if __name__ == "__main__":
    app()
