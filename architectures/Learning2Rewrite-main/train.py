import os
import json
import torch
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    TrainerCallback
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

import wandb
wandb.init(project="finetune_llama", name="finetune_final")


THRESHOLD = 1.6  # Threshold for loss calibration

batch_size = 1
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.bos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


domains = ["AcademicResearch", "Code", "Entertainment", "GovernmentPublic", "NewsArticle", "Religious", "ArtCulture", "Environmental", "LegalDocument", "OnlineContent", "Sports", "Business", "Finance", "LiteratureCreativeWriting", "PersonalCommunication", "TechnicalWriting", "EducationMaterial", "FoodCusine", "MedicalText", "ProductReview", "TravelTourism"]

pair_data = []
for domain in domains:
    with open(f'/home/rl3424/dataset1/{domain}/human_train.json', 'r') as file:
        human = json.load(file)
    with open(f'/home/rl3424/dataset1/{domain}/AI_train.json', 'r') as file:
        GPT = json.load(file)
    
    for i in range(len(human)):
        pair_data.append({'input': human[i]['input'], 'ai': 0})
    for i in range(len(GPT)):
        pair_data.append({'input': GPT[i]['input'], 'ai': 1})

dataset = Dataset.from_list(pair_data)
dataset = dataset.map(lambda examples: {
    'input': [
        tokenizer.batch_decode(tokenizer.apply_chat_template([
            {
                "role": "system", 
                "content": "You are a helpful assistant. Give your output directly.",
            },
            {
                "role": "user", 
                "content": tokenizer.decode(
                    tokenizer.encode(f'''Refine this for me please: {x}''', truncation=True, max_length=1500), 
                    skip_special_tokens=True)
            },
            {
                "role": "assistant", 
                "content": tokenizer.decode(
                    tokenizer.encode(x, truncation=True, max_length=1500), 
                    skip_special_tokens=True)
            },
        ], return_tensors="pt"))[0]
        for x in examples['input']],
}, batched=True)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


class MyLlama(torch.nn.Module):
    def __init__(self):
        super(MyLlama, self).__init__()
        self.llama = model

    def forward(self, input_ids, attention_mask, labels, ai_label):       
        result = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        # Shift so that tokens < n predict n
        shift_logits = result['logits'][..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        
        # Mask out the loss for the prompt
        # Convert the input IDs to a list to find the sequence of [/INST] tokens
        input_ids_list = shift_labels.tolist()
        loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)
        # Function to find the sequence of token IDs in the input
        def find_token_sequence(seq, token_sequence):
            n = len(token_sequence)
            for i in range(len(seq) - n + 1):
                if seq[i:i + n] == token_sequence:
                    return i + n - 1  # Return the end position of the token sequence
            return -1
        # Find the positions of the [/INST] sequence and create a mask
        for batch_idx, seq in enumerate(input_ids_list):
            inst_position = find_token_sequence(seq, [518, 29914, 25580, 29962])
            if inst_position != -1:
                loss_mask[batch_idx, :inst_position + 1] = False
        # Apply the mask to shift_labels before flattening
        shift_labels[~loss_mask] = -100
        
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        shift_logits = shift_logits.view(-1, self.llama.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        
        # Compute the loss
        loss = loss_fct(shift_logits, shift_labels)
        # Add sign to loss - AI ones should have lower loss, and human ones higher loss
        ai_label = ai_label.unsqueeze(1).repeat(1, input_ids.size(1) - 1).view(-1)
        
        # # No calibration
        # modified_loss = torch.where(ai_label == 0, -loss, loss)
        # result['loss'] = modified_loss.mean()
        
        # Calibration with a hard coded threshold
        mean_loss = loss.mean()
        if (mean_loss < THRESHOLD and ai_label[0] == 0) or (mean_loss > THRESHOLD and ai_label[0] == 1):
            modified_loss = torch.where(ai_label == 0, -loss, loss)
            result['loss'] = modified_loss.mean()
        else:
            result['loss'] = torch.tensor(0., requires_grad=True, device="cuda:0")
        
        return result


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_non_packed_dataloader(
            self,
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            formatting_func=None,
            add_special_tokens=True,
            remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            # Add ai_label field to assign the sign of the loss
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"],
                    "ai_label": element["ai"]}
        
        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        # Must remove original dataset columns
        tokenized_dataset = tokenized_dataset.remove_columns(dataset.column_names)
        return tokenized_dataset

    def training_step(self, model, inputs):
        model.llama.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


# Define a custom callback to log the absolute value of the loss
class AbsoluteLossCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'loss' in state.log_history[-1]:
            loss = state.log_history[-1]['loss']
            wandb.log({'abs_train_loss': abs(loss), 'step': state.global_step})
        # Log evaluation loss
        if 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1]['eval_loss']
            wandb.log({'abs_train_loss': abs(eval_loss), 'step': state.global_step})


# Define training arguments
training_args = SFTConfig(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-6,
    do_eval=True,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    remove_unused_columns=False,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,  # Keep logging every 10 steps
    report_to="wandb",  # Integrate W&B
    gradient_accumulation_steps=32,   # accumulate gradients
    # lr_scheduler_type="linear",  # Adding learning rate scheduler
    # warmup_steps=100,  # Add warmup steps if necessary
)

model=MyLlama()

# Create the SFT trainer
trainer = CustomSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    max_seq_length=1024,
    packing=False,
    args=training_args,
    dataset_text_field="input",
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # peft_config=lora_config,
    # callbacks=[AbsoluteLossCallback()]
)

# Train the model (actually optimizing inputs)
trainer.train()

model.llama.save_pretrained("./models/llama-3-8b/")
tokenizer.save_pretrained("./models/llama-3-8b/")