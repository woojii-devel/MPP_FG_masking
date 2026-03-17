import torch
class PaddingCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        max_len = max(len(x) for x in input_ids)

        padded_inputs = []
        padded_masks = []

        for x in input_ids:
            pad_len = max_len - len(x)
            padded_inputs.append(
                torch.cat([x, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            padded_masks.append(
                torch.cat([torch.ones(len(x)), torch.zeros(pad_len)])
            )

        batch = {
            "input_ids": torch.stack(padded_inputs),
            "attention_mask": torch.stack(padded_masks),
            "labels": torch.stack(labels),
        }

        return batch