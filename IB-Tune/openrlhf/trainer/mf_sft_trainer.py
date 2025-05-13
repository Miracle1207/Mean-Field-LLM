import os
from abc import ABC
import math

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.optim import Optimizer
from tqdm import tqdm
torch.cuda.synchronize()
from openrlhf.models import GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from .mean_field_utils import calculate_loss_batch
import time

class MFTrainer(ABC):
    """
    Trainer for mean field fine-tuning.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)
        
        if self.args.alg == "mf":   # step 1: fine tune mean field model
            self.mf_model = model
            print(f"mf_model initialized as a pretrain_model")
        elif self.args.have_mf_model == True and self.args.alg == "policy_on_mf":   # step 2: train policy model based on mean field model;
            MODEL_PATH = "" # todo
            MODEL_NAME = "" # todo: you need load your tuned mean field model; or testing pretrain LLM is ok
            model_name = MODEL_PATH + MODEL_NAME
            self.mf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            print(f"mf_model:{MODEL_NAME}")
        else:
            self.mf_model = None

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            import stat
            self._wandb = wandb

            wandb_dir = os.path.abspath("./wandb_logs")
            # 检查目录是否存在
            if not os.path.exists(wandb_dir):
                try:
                    # 创建目录
                    os.makedirs(wandb_dir)
                    print(f"目录 {wandb_dir} 已创建。")
                except Exception as e:
                    print(f"创建目录 {wandb_dir} 时发生错误: {e}")
                    exit(1)

            # 修改目录权限为可写
            try:
                os.chmod(wandb_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)  # 755权限
                print(f"目录 {wandb_dir} 权限已设置为可写。")
            except Exception as e:
                print(f"修改目录 {wandb_dir} 权限时发生错误: {e}")
                exit(1)

            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                # mode="offline",
                dir=wandb_dir,
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def cosine_alpha(self, step, total_steps, alpha_max):
        """
        Cosine schedule for alpha (mf_coef).
        Args:
            step: Current training step.
            total_steps: Total number of training steps.
            alpha_max: Maximum value for alpha (mf_coef).
        Returns:
            Alpha value for the current step.
        """
        if step >= total_steps:
            return alpha_max
        return 0.5 * alpha_max * (1 - math.cos(math.pi * step / total_steps))

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        """
        Modified fit() function with cosine alpha scheduling.
        """
        # Set evaluation and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # Do not save checkpoint
    
        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)
    
        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
        
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )
        
            self.model.train()
            for step_idx, batch in enumerate(self.train_dataloader):
                if len(batch) == 0:  # 或 len(batch) == 0
                    print(f"[step {step_idx}] Empty batch encountered, skip and resample.")
                    mean_field = ["暂无整体信息"]
                    continue

                try:
                    batch_s, batch_a, batch_idx = zip(*batch)

                except ValueError as e:
                    print(f"[step {step_idx}] Batch unpack error: {e}, skip and resample.")
                    continue

                s_t = list(batch_s)
                a_star_t = list(batch_a)
                if batch_idx[-1] < 6*len(batch) or args.use_true_action:  # 前面
                    use_true_action = True
                else:
                    use_true_action = False

                if step_idx == 0:
                    mean_field = ""
                    next_mf_loss = torch.tensor(0.)

                mf_loss = next_mf_loss
                action_loss,policy_loss, next_mf_loss, mean_field = calculate_loss_batch(
                    alg=args.alg,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    states=s_t,
                    actions_true=a_star_t,
                    previous_mf=mean_field,
                    use_true_action=use_true_action,
                    mf_model=self.mf_model.to(torch.cuda.current_device()),
                    device=torch.cuda.current_device(),
                )
                if self.args.alg == "mf":
                    loss = 2* action_loss + mf_loss
                elif self.args.alg == "policy_on_mf":
                    loss = policy_loss
                else:
                    loss = action_loss

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                if step_idx == 0:
                    total_loss_mean = loss.item()
                    loss_mean = action_loss.item()
                else:
                    total_loss_mean = total_loss_mean * 0.9 + 0.1 * loss.item()
                    loss_mean = loss_mean * 0.9 + 0.1 * action_loss.item()
            
                logs_dict = {
                    "gpt_loss": action_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "total_loss": loss.item(),
                    "mf_loss": mf_loss.item(),
                    "loss_mean": loss_mean,
                    "total_loss_mean": total_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                step_bar.set_postfix(logs_dict)
                step_bar.update()
                if step % 10 == 0:
                    print("mean field:", mean_field)
            
                # Save logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                step += 1
        
            epoch_bar.update()
    
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        """
        Evaluate the model on the evaluation dataloader.

        Args:
            eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
            steps (int): Current step (optional).
        """
        self.model.eval()
        loss_mean = 0
        mean_field = ""
        next_mf_loss = torch.tensor(0.)# Initialize mean_field for evaluation
        times = 0
    
        with torch.no_grad():
            step_bar = tqdm(
                range(len(eval_dataloader)),
                desc=f"Eval stage at step {steps}",
                disable=not self.strategy.is_rank_0(),
            )
        
            for batch in eval_dataloader:
                if len(batch) == 0:  # 或 len(batch) == 0
                    print(f"[step {times}] Empty batch encountered, skip and resample.")
                    mean_field = ["暂无整体信息"]
                    continue
    
                try:
                    batch_s, batch_a, batch_idx = zip(*batch)
    
                except ValueError as e:
                    print(f"[step {times}] Batch unpack error: {e}, skip and resample.")
                    continue
    
                s_t = list(batch_s)
                a_star_t = list(batch_a)
                if batch_idx[-1] < 6 * len(batch) or self.args.use_true_action:  # 前面
                    use_true_action = True
                else:
                    use_true_action = False
    
                mf_loss = next_mf_loss
                action_loss,policy_loss, next_mf_loss, mean_field = calculate_loss_batch(
                    alg=self.args.alg,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    states=s_t,
                    actions_true=a_star_t,
                    previous_mf=mean_field,
                    use_true_action=use_true_action,
                    mf_model=self.mf_model.to(torch.cuda.current_device()),
                    device=torch.cuda.current_device(),
                )
                if self.args.alg == "mf":
                    loss = 2*action_loss + mf_loss
                elif self.args.alg == "policy_on_mf":
                    loss = policy_loss
                else:
                    loss = action_loss
        
                if times == 0:
                    total_loss_mean = loss.item()
                    loss_mean = action_loss.item()
                else:
                    total_loss_mean = total_loss_mean * 0.9 + 0.1 * loss.item()
                    loss_mean = loss_mean * 0.9 + 0.1 * action_loss.item()
    
                logs_dict = {
                    "gpt_loss": action_loss.item(),
                    "policy_loss": policy_loss.item(),
                    "total_loss": loss.item(),
                    "mf_loss": -mf_loss.item(),
                    "loss_mean": loss_mean,
                    "total_loss_mean": total_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                
            
                # Update progress bar with current metrics
                step_bar.set_postfix(logs_dict)
                step_bar.update()
            
                times += 1
        
            # Log evaluation metrics to Weights & Biases or TensorBoard
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {f"eval/{k}": v for k, v in {**logs_dict, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
    
        self.model.train()

        