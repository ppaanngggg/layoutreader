from collections import OrderedDict, defaultdict
from typing import List, Dict

import torch
from transformers import LayoutLMv3ForTokenClassification

MAX_LEN = 510
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2


class DataCollator:
    def __init__(self, skip_input_ids: bool, hidden_size: int):
        self.skip_input_ids = skip_input_ids
        self.hidden_size = hidden_size

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        bbox = []
        labels = []
        input_ids = []
        attention_mask = []

        # clip bbox and labels to max length, build input_ids and attention_mask
        for feature in features:
            _bbox = feature["source_boxes"]
            if len(_bbox) > MAX_LEN:
                _bbox = _bbox[:MAX_LEN]
            _labels = feature["target_index"]
            if len(_labels) > MAX_LEN:
                _labels = _labels[:MAX_LEN]
            _input_ids = [UNK_TOKEN_ID] * len(_bbox)
            _attention_mask = [1] * len(_bbox)
            assert len(_bbox) == len(_labels) == len(_input_ids) == len(_attention_mask)
            bbox.append(_bbox)
            labels.append(_labels)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)

        # add CLS and EOS tokens
        for i in range(len(bbox)):
            bbox[i] = [[0, 0, 0, 0]] + bbox[i] + [[0, 0, 0, 0]]
            labels[i] = [-100] + labels[i] + [-100]
            input_ids[i] = [CLS_TOKEN_ID] + input_ids[i] + [EOS_TOKEN_ID]
            attention_mask[i] = [1] + attention_mask[i] + [1]

        # padding to max length
        max_len = max(len(x) for x in bbox)
        for i in range(len(bbox)):
            bbox[i] = bbox[i] + [[0, 0, 0, 0]] * (max_len - len(bbox[i]))
            labels[i] = labels[i] + [-100] * (max_len - len(labels[i]))
            input_ids[i] = input_ids[i] + [EOS_TOKEN_ID] * (max_len - len(input_ids[i]))
            attention_mask[i] = attention_mask[i] + [0] * (
                max_len - len(attention_mask[i])
            )

        ret = {
            "bbox": torch.tensor(bbox),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
        if self.skip_input_ids:
            ret["inputs_embeds"] = torch.zeros(
                (len(features), max_len, self.hidden_size)
            )
        else:
            ret["input_ids"] = torch.tensor(input_ids)
        # set label > MAX_LEN to -100, because original labels may be > MAX_LEN
        ret["labels"][ret["labels"] > MAX_LEN] = -100
        # set label > 0 to label-1, because original labels are 1-indexed
        ret["labels"][ret["labels"] > 0] -= 1
        return ret


def prepare_inputs(
    inputs: Dict[str, torch.Tensor], model: LayoutLMv3ForTokenClassification
) -> Dict[str, torch.Tensor]:
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret


def parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    """
    parse logits to orders

    :param logits: logits from model
    :param length: input length
    :return: orders
    """
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=True).tolist()
    ret = OrderedDict()
    for order in orders:
        for idx in order:
            if idx not in ret:
                ret[idx] = None
                break
    return list(ret.keys())


def parse_logits_v2(logits: torch.Tensor, length: int) -> List[int]:
    """
    parse logits to orders

    :param logits: logits from model
    :param length: input length
    :return: orders
    """
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        # filter idxes len > 1
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        # filter
        for order, idxes in order_to_idxes.items():
            # find original logits of idxes
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            # keep the highest logit as order, set others to next candidate
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()

    return ret


def check_duplicate(a: List[int]) -> bool:
    return len(a) != len(set(a))
