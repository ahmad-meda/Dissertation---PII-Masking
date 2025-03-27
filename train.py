import os
import srsly

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.utils import load_config_as_namespace
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing import WordsSplitter, GLiNERDataset
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator

device = torch.device('cuda:0')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train      = srsly.read_json('data/train.json')
validation = srsly.read_json('data/validation.json')

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(device)
data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)

# calculate number of epochs
batch_size = 32
num_epochs = 3

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,
    others_weight_decay=0.01,
    lr_scheduler_type="cosine", #linear
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    eval_steps=1500,
    logging_steps=150,
    save_steps = 1500,
    fp16=True,
    save_total_limit=10,
    dataloader_num_workers = 4,
    use_cpu = False,
    report_to="none",
    push_to_hub=True,
    hub_model_id="Ahmad-Meda/gliner_multi-300k-v1",
    hub_token="your_hub_token",
    save_only_model=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=validation,
    tokenizer=model.data_processor.transformer_tokenizer,
    data_collator=data_collator,
)

trainer.train()