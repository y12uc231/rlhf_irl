# Reference : EleutharAI PPO code scripts

import json
import math
import os
import sys
from itertools import islice

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tritonclient.utils import np_to_triton_dtype


import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import wandb

wandb.login()

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10000,
        total_steps=10000,
        batch_size=4,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        entity_name = 'eleutherai',
        project_name = 'pythia-rlhf',
        save_best = False
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-70m", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-70m", truncation_side="left", padding_side = "left", ),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=128,
            top_k=30,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

config_name = os.environ.get("CONFIG_NAME", "70M")
if config_name == "70M":
    # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/do2vbz2o
    default_config.train.batch_size = 8
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 750
    default_config.model.model_path = "lomahony/eleuther-pythia70m-hh-sft"
    default_config.model.num_layers_unfrozen = 4
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-70m/"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-70m"
    default_config.optimizer.kwargs["lr"] = 5.45e-6
    default_config.optimizer.kwargs["weight_decay"] = 0.0006
    default_config.scheduler.kwargs["eta_min"] = 5.45e-6
    default_config.method.num_rollouts = 32
    default_config.method.target = 5.71
    default_config.method.ppo_epochs = 8
    default_config.method.chunk_size = 4
    default_config.train.group_name = "EleutherAI/pythia-70m-ppo"
elif config_name == "160M":
    # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/jubaluv8
    default_config.train.batch_size = 8 
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 750
    default_config.model.model_path = "lomahony/eleuther-pythia160m-hh-sft"
    default_config.model.num_layers_unfrozen = 4
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-160m/"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-160m"
    default_config.optimizer.kwargs["lr"] = 1.7e-6
    default_config.optimizer.kwargs["weight_decay"] = 3.81e-5
    default_config.scheduler.kwargs["eta_min"] = 1.7e-6
    default_config.method.num_rollouts = 48
    default_config.method.chunk_size = 4
    default_config.method.ppo_epochs = 6
    default_config.method.target = 6.42 
    default_config.train.group_name = "EleutherAI/pythia-160m"
elif config_name == "410M":
    # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/vpuhppgx
    default_config.train.batch_size = 8
    default_config.train.total_steps = 750
    default_config.train.seq_length = 1024
    default_config.model.num_layers_unfrozen = 3
    default_config.optimizer.kwargs["lr"] = 2.2e-7
    default_config.scheduler.kwargs["eta_min"] = 2.2e-7
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-410m"
    default_config.model.model_path = "lomahony/eleuther-pythia410m-hh-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-410m"
    default_config.method.chunk_size = 4
    default_config.method.num_rollouts = 48
    default_config.method.ppo_epochs = 5
    default_config.method.target = 4.9
    default_config.train.group_name = "EleutherAI/pythia-410m"
elif config_name == "1B":
    default_config.train.batch_size = 4
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 2500
    default_config.optimizer.kwargs["lr"] = 6e-6
    default_config.optimizer.kwargs["weight_decay"] = 0.0002
    default_config.scheduler.kwargs["eta_min"] = 6e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_1B"
    default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
    default_config.method.chunk_size = 16
    default_config.train.group_name = default_config.model.model_path
elif config_name == "1.4B":
    default_config.train.batch_size = 4
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 1500
    default_config.model.num_layers_unfrozen = 2
    default_config.optimizer.kwargs["lr"] = 6e-6
    default_config.scheduler.kwargs["eta_min"] = 6e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-1.4b"
    default_config.model.model_path = "lomahony/eleuther-pythia1.4b-hh-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
    default_config.method.chunk_size = 4
    default_config.method.num_rollouts = 48
    default_config.method.ppo_epochs = 8
    default_config.method.target = 5.067
    default_config.train.group_name = "EleutherAI/pythia-1.4b-ppo-hptuning"
elif config_name == "2.8B":
    # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/9ui2aa0x
    default_config.train.batch_size = 2
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 3000
    default_config.model.num_layers_unfrozen = 19
    default_config.optimizer.kwargs["lr"] = 2.3e-6
    default_config.optimizer.kwargs["weight_decay"] = 1.7e-3
    default_config.scheduler.kwargs["eta_min"] = 2.3e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh/pythia-2.8b"
    default_config.model.model_path = "lomahony/eleuther-pythia2.8b-hh-sft"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-2.8b"
    default_config.method.chunk_size = 4
    default_config.method.num_rollouts = 48
    default_config.method.ppo_epochs = 4
    default_config.method.target = 5
    default_config.train.group_name = "EleutherAI/pythia-2.8b-ppo"
elif config_name == "6.9B":
    # Following params from https://wandb.ai/eleutherai/pythia-rlhf/runs/2uue7ywj
    default_config.train.batch_size = 1
    default_config.train.seq_length = 1024
    default_config.train.total_steps = 6000
    default_config.train.checkpoint_dir = "/fsx/orz/pythia-rlhf-checkpoints/ppo_hh/pythia-6.9b"
    default_config.model.model_path = "lomahony/eleuther-pythia6.9b-hh-sft"
    default_config.model.num_layers_unfrozen = 2
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-6.9b"
    default_config.optimizer.kwargs["lr"] = 3.32e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.method.num_rollouts = 48
    default_config.method.chunk_size = 4
    default_config.method.ppo_epochs = 3
    default_config.method.target = 5.067
    default_config.train.group_name = "EleutherAI/pythia-6.9b-ppo"
elif config_name == "12B":
    default_config.train.batch_size = 1
    default_config.train.seq_length = 768
    default_config.train.total_steps = 5000
    default_config.optimizer.kwargs["lr"] = 1e-6
    default_config.scheduler.kwargs["eta_min"] = 1e-6
    default_config.train.checkpoint_dir = "checkpoints/ppo_hh_pythia_12b"
    default_config.model.model_path = "EleutherAI/pythia-12b"
    default_config.tokenizer.tokenizer_path = "EleutherAI/pythia-12b"
    default_config.method.num_rollouts = 32
    default_config.method.chunk_size = 8
    default_config.train.group_name = default_config.model.model_path
else:
    raise ValueError(f"Config {config_name} does not exist."
    "Please append this config into the file before calling it")
    


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def create_reward_fn():  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    triton_host = os.environ.get("TRITON_HOST")

    if triton_host:
        triton_url, triton_model = triton_host.split("/")
        client = client_util.InferenceServerClient(url=triton_url, verbose=False)

        def reward_fn(samples, prompts, outputs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            input = reward_tokenizer(samples, padding=True, max_length=1024)

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

                result = client.infer(triton_model, [prepare_tensor("input_ids", input_ids)])
                rewards = result.as_numpy("rewards")
                out.extend(rewards)

            return out

    elif os.environ.get("RANK", "0") == "0":
        
        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path, eos_token_id):
                super().__init__()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                self.transformer = model.transformer
                self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
                self.eos_token_id = eos_token_id

            def forward(self, input_ids):
                states = self.transformer(input_ids)[0]
                rewards = self.v_head(states).squeeze(-1)
                ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
                returns = torch.gather(rewards, 1, ends).squeeze(-1)
                return returns

        reward_model = RewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
        directory = snapshot_download("Dahoas/gptj-rm-static", revision="676bfd4d")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(reward_device)
        reward_batch_size = 24
        delta_reward = True

        def get_reward(samples):
            input = reward_tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(reward_device)

            mbs = reward_batch_size
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = reward_model(input_ids)
                out.extend(rewards)
            return torch.hstack(out)

        def reward_fn(samples, prompts, original_output, **kwargs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            rewards = get_reward(samples)

            if not delta_reward:
                return rewards

            original_samples = [p + o + reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards

    else:
        reward_fn = True

    return reward_fn


def main(hparams={}):
    trlx.logging.set_verbosity(trlx.logging.INFO)
    config = TRLConfig.update(default_config, hparams)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    
    # Change to toxicity
    dataset = load_dataset("Dahoas/full-hh-rlhf")
    prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 35)]


    # Change reward model to be facebook/roberta-hate-speech-dynabench-r4-target
    reward_fn = create_reward_fn()

    trainer, eval_stats = trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )
    if trainer.accelerator.is_main_process:
        trainer.accelerator.print("\n"*100)
        trainer.accelerator.print(eval_stats["reward/mean"])

if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    if len(sys.argv) > 2:
        default_config.optimizer.kwargs['lr'] = float(sys.argv[2])
        default_config.scheduler.kwargs['eta_min'] = float(sys.argv[2])
        default_config.optimizer.kwargs['weight_decay'] = float(sys.argv[3])
    # main(hparams)
    main(hparams)


