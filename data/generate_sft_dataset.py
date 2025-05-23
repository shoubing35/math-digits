import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    # script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # # remove output_dir if exists
    # shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # # Set seed for reproducibility
    # import random
    # import numpy as np
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)  # Enforce determinism
    #
    # ################
    # # Model & Tokenizer
    # ################
    # torch_dtype = (
    #     model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    # )
    # quantization_config = get_quantization_config(model_args)
    # model_kwargs = dict(
    #     revision=model_args.model_revision,
    #     attn_implementation=model_args.attn_implementation,
    #     torch_dtype=torch_dtype,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    # )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    #
    # print("Vocab size:", tokenizer.vocab_size)  # charles debug
    #
    # value_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    # )
    # policy = AutoModelForCausalLM.from_pretrained(
    #     training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    # )
    #
    # print("Value model vocab size:", value_model.config.vocab_size)  # charles debug
    # print("Reward model vocab size:", reward_model.config.vocab_size)  # charles debug
    # print("Policy model vocab size:", policy.config.vocab_size)  # charles debug
    #
    # peft_config = get_peft_config(model_args)
    # if peft_config is None:
    #     ref_policy = AutoModelForCausalLM.from_pretrained(
    #         training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    #     )
    # else:
    #     ref_policy = None

    # charles + gpt:
    # Part 1: Generate 5 Prompts and Create 10 Pairwise Comparisons in a CSV
    import itertools
    import pandas as pd
    # from peft import get_peft_model
    import torch

    # peft_model = get_peft_model(policy, peft_config)

    # Generate 1 example
    from datasets import Dataset, DatasetDict
    df = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/my_dataset/sft_dataset.csv")
    df["text"] = df["text_instr"] + "\n" + df["text_input"] + "\n" + df["text_label"]
    df["text_prompt"] = df["text_instr"] + "\n" + df["text_input"] + "\n"
    # df = df[["text"]]  # keep only the 'text' column

    # Optional: Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create train/test/val split: 80/10/10
    train_end = int(0.8 * len(df))
    val_end = int(0.9 * len(df))

    ds = DatasetDict({
        "train": Dataset.from_pandas(df[:train_end]),
        "val": Dataset.from_pandas(df[train_end:val_end]),
        "test": Dataset.from_pandas(df[val_end:])
    })

    ds.push_to_hub("shoubing35/ones_digit_sft_dataset")

