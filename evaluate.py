import srsly

import torch
from gliner import GLiNER
from datasets import load_dataset

train = load_dataset("ai4privacy/pii-masking-300k", split="train")

entity_types = list(set(label[-1] for labels in train["span_labels"] for label in srsly.json_loads(labels)))

model = GLiNER.from_pretrained("Ahmad-Meda/gliner_multi-300k-v1").to('cuda:0')

eval_data = srsly.read_json('data/validation.json')

evaluation_results = model.evaluate(
    eval_data, flat_ner=True, entity_types=entity_types, batch_size=16
)

print(evaluation_results[0])