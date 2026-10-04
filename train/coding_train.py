"""
Policy / Reward / Environment model trainer for the coding agentic-RL pipeline
(forked from the VICO alfworld trainer).

- PPO-clip surrogate on step-level advantages (reward/llm_rl_reward.py) with a
  K3-estimated KL penalty (use_kl_estimator_k3) to keep updates small and avoid
  loss spikes / garbled generations.
- DeepSpeed ZeRO-3 + bf16 through accelerate (accelerate_configs/*zero3.yaml).
- training.target in {policy, reward, env}.
- Response-only logits (`logits_to_keep`) + chunked fp32 log-softmax: avoids materializing
  (L x |V|) logits for long multi-step prompts, which is the real memory peak for 4B models.
- Per-batch padding trim (multi-step prompts are long and have very different
  lengths), per-step KL / clip-fraction / loss logged to <project>/logs/train_metrics.jsonl.
- Crash-safe checkpoint: written to a tmp dir and swapped in with os.replace,
  so a crash during saving never corrupts ckpt/<name>; snapshot every save_every epochs.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from torch.utils.data import Dataset, DataLoader

from train.prompting_utils import UniversalPrompting
from train.lr_schedulers import get_scheduler
from train.log_utils import set_verbosity_info, set_verbosity_error
from train.utils import get_config, flatten_omega_conf, AverageMeter







try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")







class TrainDataset(Dataset):
    def __init__(self, input_ids, p_mask, labels, advantage):
        self.input_ids = input_ids           # (N, L)
        self.p_mask    = p_mask              # (N, L) bool; True on response tokens
        self.labels    = labels              # (N, L) same as input_ids (we'll shift when computing logp)
        self.advantage = advantage           # list[float] length N (group-centered rewards)

        # old logp cache, aligned to p_mask shape (we fill positions 1: only)
        self.logp_old_tok = torch.full((len(input_ids), p_mask.shape[1]), float('-inf'))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.input_ids[idx],
            self.p_mask[idx],
            self.labels[idx],
            self.advantage[idx],
        )



def main():
    
    config = get_config()

    project_name = config.experiment.project

    def _resolve(base_path, opt_name):
        ckpt = "./" + project_name + "/ckpt/" + opt_name
        if int(config.experiment.current_epoch) > 1 and os.path.isdir(ckpt):
            return ckpt
        return base_path

    if config.training.target == "policy":
        pretrained_model = _resolve(config.model.policy_model, config.model.optimized_name)
        optimized_name = config.model.optimized_name
        max_prompt_len = config.training.policy.max_prompt_len
        max_gen_length = config.training.policy.max_gen_length
        optimization_data = config.dataset.optimization_data
        update_per_step = config.training.policy.update_per_step
        batch_size_lm = config.training.policy.batch_size_lm
        gradient_checkpointing_enable = config.training.policy.gradient_checkpointing_enable
    elif config.training.target == "reward":
        pretrained_model = _resolve(config.model.reward_model, config.model.optimized_reward_name)
        optimized_name = config.model.optimized_reward_name
        max_prompt_len = config.training.reward.max_prompt_len
        max_gen_length = config.training.reward.max_gen_length
        optimization_data = config.dataset.reward_optimization_data
        update_per_step = config.training.reward.update_per_step
        batch_size_lm = config.training.reward.batch_size_lm
        gradient_checkpointing_enable = config.training.reward.gradient_checkpointing_enable
    elif config.training.target == "env":
        pretrained_model = _resolve(config.model.environment_model, config.model.optimized_env_name)
        optimized_name = config.model.optimized_env_name
        max_prompt_len = config.training.env.max_prompt_len
        max_gen_length = config.training.env.max_gen_length
        optimization_data = config.dataset.env_optimization_data
        update_per_step = config.training.env.update_per_step
        batch_size_lm = config.training.env.batch_size_lm
        gradient_checkpointing_enable = config.training.env.gradient_checkpointing_enable
    else:
        raise ValueError(f"unknown training.target={config.training.target}")

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False



    ##################################
    #         DATALOADER             #
    #################################


    def simple_collate(batch):
        ids, input_ids, p_mask, labels, adv = zip(*batch)
        return {
            "ids":        torch.tensor(ids),
            "input_ids":  torch.stack(input_ids),
            "p_mask":     torch.stack(p_mask),
            "labels":     torch.stack(labels),
            "advantage":  torch.tensor(adv, dtype=torch.float32),
        }



    
    data_path = "./" + project_name + "/temp_data/" + optimization_data + ".json"
    dataset_load = []
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            dataset_load = json.load(f)
    # e.g. env model at epoch 1 (its rewards arrive one epoch later) or every group had zero advantage
    if len(dataset_load) == 0:
        print(f"[train:{config.training.target}] no training data in {data_path}, skip this update.", flush=True)
        return

    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=max_prompt_len,
                                       max_gen_length=max_gen_length,
                                       ignore_id=-100)
    # Filter over-long prompts HERE (together with their rewards). UniversalPrompting drops them
    # internally without returning the kept indices, which would misalign rewards and samples.
    prompt_lens = tokenizer(prompt_list, padding=False, truncation=False, return_length=True)["length"]
    keep = [i for i, L in enumerate(prompt_lens) if L <= max_prompt_len]
    drop_num = len(prompt_list) - len(keep)
    prompt_list = [prompt_list[i] for i in keep]
    response_list = [response_list[i] for i in keep]
    reward_list = [reward_list[i] for i in keep]
    dataset_load = [dataset_load[i] for i in keep]
    if len(keep) == 0:
        print(f"[train:{config.training.target}] all {drop_num} samples exceed max_prompt_len, skip.", flush=True)
        return
    input_ids_lm, _, start_pos, _ = uni_prompting((prompt_list, response_list))






    def build_grpo_dataset(input_ids_lm, start_pos, rewards, dataset_load, tokenizer, config):
        """
        Groups items by (group_id if present) else by exact prompt string,
        computes group-relative advantages (reward - baseline),
        and returns tensors ready for TrainDataset.
        """
        B, L = input_ids_lm.shape
        pad_id = tokenizer.pad_token_id

        # p_mask: response tokens only (and not padding)
        p_mask = torch.zeros((B, L), dtype=torch.bool)
        p_mask[:, start_pos:] = True
        p_mask &= (torch.as_tensor(input_ids_lm) != pad_id)

        input_ids = torch.as_tensor(input_ids_lm, dtype=torch.long)
        labels    = input_ids.clone()   # we'll shift when we compute logp


        return input_ids, p_mask, labels, rewards
        

    
    # Build GRPO dataset in one shot (no extensions/duplication)
    input_ids, p_mask, labels, advantages = build_grpo_dataset(
        input_ids_lm, start_pos, reward_list, dataset_load, tokenizer, config
    )
    dataset_lm = TrainDataset(input_ids, p_mask, labels, advantages)



    #########################
    # SETUP Accelerator     #
    #########################

    ws = int(os.environ.get("WORLD_SIZE", "1")) 
    
    gradient_accumulation_steps = max(1, math.ceil(len(dataset_lm) / (update_per_step * batch_size_lm * ws)))
    config.experiment.logging_dir = str(Path(project_name) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )
    assert ws == accelerator.num_processes

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=project_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            project_name,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(project_name, exist_ok=True)
        config_path = Path(project_name) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    
    

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")

    

    if gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    
    

    

    total_batch_size_lm = batch_size_lm * ws * gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, math.ceil(len(dataset_lm) / total_batch_size_lm))
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )





    

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )





    import torch.nn.functional as F


    def trim_cols(input_ids, pad_id):
        """Column range [lo, hi) that contains every non-pad token of the batch
        (prompts are left-padded, responses right-padded to the dataset max)."""
        nonpad = (input_ids != pad_id).any(dim=0)
        idx = torch.nonzero(nonpad, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 0, input_ids.size(1)
        return int(idx[0].item()), int(idx[-1].item()) + 1

    LOGP_CHUNK = int(config.training.get("logp_chunk_size", 1024))

    def response_token_logps(input_ids, attention_mask, position_ids, p_mask):
        """
        Per-token log-prob of the RESPONSE tokens only, aligned to labels_shifted (B, L-1).

        Memory: a full forward materializes logits of shape (B, L, |V|); with L≈12k and |V|≈152k
        that alone is several GB (plus log_softmax and its grad). Responses sit at the END of the
        sequence (prompt left-padded, response right-padded), so we ask the HF model for the last K
        positions only (`logits_to_keep`, K ≈ max response length + 1) and compute log-softmax in fp32
        chunk by chunk (fp32 also makes the PPO ratio numerically more accurate than bf16).
        Positions outside the response are returned as 0 and are masked out by p_mask later.
        """
        B, L = input_ids.shape
        cols = torch.nonzero(p_mask.any(dim=0), as_tuple=False).squeeze(-1)
        first = max(1, int(cols[0].item())) if cols.numel() > 0 else L - 1   # first response column
        K = L - first + 1                                                    # positions first-1 .. L-1
        try:
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, logits_to_keep=K).logits
        except TypeError:                                                    # older transformers
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids).logits
        logits = logits[:, -K:, :][:, :-1, :]                                # predict tokens first .. L-1
        targets = input_ids[:, first:]                                       # (B, K-1)
        parts = []
        for s in range(0, targets.size(1), LOGP_CHUNK):
            lg = logits[:, s:s + LOGP_CHUNK, :].float()
            tg = targets[:, s:s + LOGP_CHUNK]
            parts.append(lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, dim=-1))
        logp_resp = torch.cat(parts, dim=1)                                  # (B, K-1)
        out = torch.zeros((B, L - 1), dtype=logp_resp.dtype, device=logp_resp.device)
        out = torch.cat([out[:, : first - 1], logp_resp], dim=1)             # keep autograd graph simple
        return out

    def make_attn_and_pos(input_ids, pad_id):
        # 2D attention mask
        attention_mask = (input_ids != pad_id).to(torch.long)  # (B, L)

        position_ids = attention_mask.cumsum(dim=1) - 1        # (B, L)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return attention_mask, position_ids

    @torch.no_grad()
    def compute_logp_old_tok_parallel(accelerator, dataset, train_dataloader_lm, pad_id):
        model.eval()
        import torch.nn.functional as F

        from tqdm.auto import tqdm

        dl = train_dataloader_lm
        iterator = (
            tqdm(
                dl,
                desc="Precomputing old token log-probs",
                dynamic_ncols=True,
                disable=not accelerator.is_local_main_process,
                leave=True,
            )
            if (tqdm is not None)
            else dl
        )

        for batch in iterator:
            ids       = batch["ids"]
            lo, hi    = trim_cols(batch["input_ids"], pad_id)
            input_ids = batch["input_ids"][:, lo:hi].to(accelerator.device)
            p_mask    = batch["p_mask"][:, lo:hi].to(accelerator.device)

            # standard attention mask
            attention_mask, position_ids = make_attn_and_pos(input_ids, pad_id)

            # response-only, chunked fp32 log-probs (see response_token_logps)
            logp_tok = response_token_logps(input_ids, attention_mask, position_ids, p_mask)  # (B, L-1)

            # write into dataset cache aligned with full-length mask: positions [1:] get values, [0] stays -inf
            # (so later we can safely mask with p_mask[:,1:])
            full = torch.full((logp_tok.size(0), dataset.logp_old_tok.size(1)), float('-inf'), device=logp_tok.device)
            full[:, lo + 1:hi] = logp_tok
            dataset.logp_old_tok[ids] = full.float().cpu()

        accelerator.wait_for_everyone()
        model.train()


    #################################
    #             Inference         #
    #################################
    logger.info("***** Running inference *****")

    compute_logp_old_tok_parallel(
        accelerator,
        dataset_lm,
        train_dataloader_lm,
        pad_id=pad_id,
    )





    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids_lm.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    


    
    import torch.nn.functional as F

    def forward_process(input_ids, p_mask, labels, adv, logp_old_tok):
        

        adv = adv.to(input_ids.device).detach()   # (B,)
        pad_id = tokenizer.pad_token_id

        attention_mask, position_ids = make_attn_and_pos(input_ids, pad_id)

        # same computation as the old log-prob pass -> ratio == 1 exactly on the first update
        logp_new_tok = response_token_logps(input_ids, attention_mask, position_ids, p_mask)  # (B, L-1)

        # align masks and old_logp to shifted axis
        p_mask_shift = p_mask[:, 1:]                              # (B, L-1)
        old_lp_shift = logp_old_tok[:, 1:]                        # (B, L-1)

        # PPO ratio on response tokens only
        ratio = (logp_new_tok - old_lp_shift)
        ratio = torch.where(p_mask_shift, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0)
        ratio = torch.exp(ratio)

        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

        adv_tok = adv.unsqueeze(1)  # broadcast over tokens
        surrogate = torch.min(ratio * adv_tok, clipped * adv_tok)
        surrogate = surrogate * p_mask_shift

        denom = torch.clamp(p_mask_shift.sum(dim=1), min=1)          # per-sample token count in response
        policy_loss = - (surrogate.sum(dim=1) / denom).mean()

        with torch.no_grad():
            clip_frac = (((ratio - 1.0).abs() > config.training.eps).float() * p_mask_shift).sum() / denom.sum()

        # KL penalty against the pre-update (rollout) policy.
        # K3 estimator: k3 = exp(logp_old - logp_new) - (logp_old - logp_new) - 1  (>= 0, unbiased, low variance)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        kl_val = torch.tensor(0.0, device=policy_loss.device)
        diff = torch.where(p_mask_shift, logp_new_tok - old_lp_shift, torch.zeros_like(logp_new_tok))
        if config.training.use_kl_estimator_k3:
            t = (-diff).clamp(-10.0, 10.0)
            kl_tok = t.exp() - 1.0 + diff
        else:
            kl_tok = diff
        kl_seq = (kl_tok * p_mask_shift).sum(dim=1) / torch.clamp(denom, min=1)
        kl_val = kl_seq.mean().detach()
        if config.training.beta > 0:
            kl_loss = config.training.beta * kl_seq.mean()

        stats = {"policy_loss": policy_loss.detach(), "kl": kl_val, "clip_frac": clip_frac.detach()}
        return policy_loss + kl_loss, stats







    from tqdm.auto import tqdm

    for epoch in range(num_train_epochs):
        model.train()
        progress = tqdm(train_dataloader_lm, disable=not accelerator.is_local_main_process, dynamic_ncols=True)

        for step, batch in enumerate(progress, start=1):

            lo, hi    = trim_cols(batch["input_ids"], pad_id)
            input_ids = batch["input_ids"][:, lo:hi]
            p_mask    = batch["p_mask"][:, lo:hi]
            adv       = batch["advantage"]
            ids_cpu   = batch["ids"].cpu()
            old_lp    = dataset_lm.logp_old_tok[ids_cpu][:, lo:hi].to(accelerator.device, non_blocking=True)

            loss, stats = forward_process(
                input_ids=input_ids,
                p_mask=p_mask,
                labels=None,
                adv=adv,
                logp_old_tok=old_lp.detach(),
            )

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (step % gradient_accumulation_steps) == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_val = float(accelerator.gather(loss.detach()).mean().item())
                kl_v = float(accelerator.gather(stats["kl"].reshape(1)).mean().item())
                clip_v = float(accelerator.gather(stats["clip_frac"].reshape(1)).mean().item())
                progress.set_postfix(loss=loss_val, kl=kl_v, clip=clip_v)
                if accelerator.is_main_process:
                    os.makedirs(os.path.join(project_name, "logs"), exist_ok=True)
                    with open(os.path.join(project_name, "logs", "train_metrics.jsonl"), "a") as mf:
                        mf.write(json.dumps({
                            "epoch": int(config.experiment.current_epoch), "target": str(config.training.target),
                            "update": int(step // gradient_accumulation_steps), "loss": loss_val,
                            "policy_loss": float(stats["policy_loss"].item()), "kl": kl_v, "clip_frac": clip_v,
                            "lr": float(lr_scheduler.get_last_lr()[0]),
                        }) + "\n")
                

    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(model, tokenizer, config, accelerator, f"{config.training.target}-epoch-{config.experiment.current_epoch}")

    accelerator.wait_for_everyone()
    accelerator.end_training()






def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        # write to a tmp dir first, then swap: a crash while saving never corrupts ckpt/<name>
        final_dir = save_base / name
        tmp_dir = save_base / f".{name}.tmp"
        old_dir = save_base / f".{name}.old"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        model_to_save.save_pretrained(
            tmp_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(tmp_dir))
        shutil.rmtree(old_dir, ignore_errors=True)
        if final_dir.exists():
            os.replace(final_dir, old_dir)
        os.replace(tmp_dir, final_dir)
        shutil.rmtree(old_dir, ignore_errors=True)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(config.experiment.current_epoch),
            "name": name,
        }
        with (save_base / f"metadata-{name}.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")

    















if __name__ == "__main__":
    main()
