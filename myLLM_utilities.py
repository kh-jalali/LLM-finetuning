#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import time

import transformers

logger = logging.getLogger(__name__)
global_config = None

#############################
########## Permissions ##########
#############################
model_name_to_id = {
  "bigger_model_name" : "06ad41e68cd839fb475a0c1a4ee7a3ad398228df01c9396a97788295d5a0f8bb"
}

#############################
########## LOGGING ##########
#############################
def initialize_config_and_logging(existing_config=None):
    global global_config
    global_config = build_config(existing_config)
    setup_logging(global_config)
    logger.debug("Config: " + str(yaml.dump(global_config.as_dict())))
    return global_config

def get_config():
    global global_config
    assert global_config is not None
    return global_config

def build_config(existing_config=None):
    configs = [
        # Using config library
        config.config_from_env(prefix="LLAMA", separator="_", lowercase_keys=True),
    ]

    if existing_config:
        if isinstance(existing_config, dict):
            configs.append(config.config_from_dict(existing_config))
        else:
            configs.append(existing_config)

    config_paths = get_config_paths()

    for path in reversed(config_paths):
        print("Loading builtin config from " + path)
        configs.append(config.config_from_yaml(path, read_from_file=True))

    return config.ConfigurationSet(*configs)

def get_config_paths():
    paths = []

def get_config_paths():
    paths = []

    config_name = "llama_config"
    config_base = "configs"

    base_config_path = os.path.join(config_base, config_name + ".yaml")
    if os.path.exists(base_config_path):
        paths.append(base_config_path)

    local_config_path = os.path.join(config_base, config_name + "_local.yaml")
    if os.path.exists(local_config_path):
        paths.append(local_config_path)

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, "." + config_name + ".yaml")
    if os.path.exists(home_config_path):
        paths.append(home_config_path)

    return paths

def setup_logging(arguments):
    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("smart_open").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


##########################
########## DATA ##########
##########################
# Wrapper for data load, split, tokenize for training
def tokenize_and_split_data(training_config, tokenizer):
  initialized_config = initialize_config_and_logging(training_config)
  dataset_path = initialized_config["datasets"]["path"]
  use_hf = initialized_config["datasets"]["use_hf"]
  print("tokenize", use_hf, dataset_path)
  if use_hf:
    dataset = datasets.load_dataset(dataset_path)
  else:
    dataset = load_dataset(dataset_path, tokenizer)
  train_dataset = dataset["train"]
  test_dataset = dataset["test"]
  return train_dataset, test_dataset

# Tokenize and split data
def load_dataset(dataset_path, tokenizer):
    random.seed(42)
    finetuning_dataset_loaded = datasets.load_dataset("json", data_files=dataset_path, split="train")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = training_config["model"]["max_length"]
    tokenized_dataset = finetuning_dataset_loaded.map(
        get_tokenize_function(tokenizer, max_length), # returns tokenize_function
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)
    return split_dataset

# Get function for tokenization, based on config parameters
def get_tokenize_function(tokenizer, _max_length):

  def tokenize_function(examples):
    max_length = _max_length

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    if "question" in examples and "answer" in examples:
      text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
      text = examples["input"][0] + examples["output"][0]
    else:
      text = examples["text"][0]

    # Run tokenizer on all the text (the input and the output)
    tokenized_inputs = tokenizer(
        text,

        # Return tensors in a numpy array (other options are pytorch or tf objects)
        return_tensors="np",

        # Padding type is to pad to the longest sequence in the batch (other option is to a certain max length, or no padding)
        padding=True,
    )

    # Calculate max length
    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        max_length
    )

    if tokenized_inputs["input_ids"].shape[1] > max_length:
        logger.warn(
            f"Truncating input from {tokenized_inputs['input_ids'].shape[1]} to {max_length}"
        )

    tokenizer.truncation_side = "left"

    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
    )

    tokenized_inputs["labels"] = tokenized_inputs["input_ids"]

    return tokenized_inputs
  return tokenize_function


# In[8]:





# In[ ]:




