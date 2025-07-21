import os
from typing import TYPE_CHECKING, Any, Dict, Optional
from copy import deepcopy

from helper import get_logger
from transformers import (
    TrainerCallback, TrainingArguments, 
    TrainerState, TrainerControl
)

from utils.average_meter import AverageMeter
from utils.timer import Timer
from constants import TRAINER_LOG


class TextLoggerCallback(TrainerCallback):
    def __init__(self, ) -> None:
        self.timer = Timer()
        self.monitor = AverageMeter("time-monitor")
        self.logger = get_logger("text-logger")


    def log(self, logs: Dict[str, Any], output_dir: str, is_train: bool) -> None:
        log_str = self.parse_logs(logs, is_train)

        self.logger.info(log_str)

        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(log_str + "\n")
        

    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of the initialization of the `Trainer`.
        """
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            self.logger.warning("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))


    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.timer.start()


    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", logs, **kwargs):
        # 非rank0不记录
        if not args.should_save:
            return
        
        # 添加训练时需要纪律的信息
        if not control.should_training_stop:
            elapsed_time = self.timer.since_start()
            self.monitor.update(elapsed_time)
            eta_time = self.monitor.average * (state.max_steps - state.global_step)

            # logs里已有loss、grad_norm、learning_rate、epoch和context中自定义的内容
            logs.update(dict(
                step = state.global_step,
                total_steps = state.max_steps,
                total_epoch = state.num_train_epochs,
                elapsed_time = elapsed_time,
                remaining_time = eta_time,
            ))
            logs = {k: v for k, v in logs.items() if v is not None}

        self.log(logs, args.output_dir, is_train = not control.should_training_stop)


    def get_time_tags(self, eta_time):
        
        ori_eta_time = eta_time

        days = int(eta_time // (60 * 60 * 24))
        eta_time = eta_time % (60 * 60 * 24)

        hours = int(eta_time // (60 * 60))
        eta_time = eta_time % (60 * 60)

        mins = int(eta_time // 60) 
        seconds = eta_time % 60

        if ori_eta_time > 60 * 60 * 24: # 60s * 60min * 24h
            return f"{days}days {hours}h {mins}m {seconds:.2f}s"

        elif 60 * 60 * 24 > ori_eta_time >= 60 * 60:
            return f"{hours}h {mins}m {seconds:.2f}s"
        
        elif 60 * 60 > ori_eta_time >= 60:
            return f"{mins}m {seconds:.2f}s"
        
        else:
            return f"{seconds:.2f}s"


    def parse_logs(self, logs: dict, is_train):
        logs = deepcopy(logs)
        
        log_str, logs_item = "", []

        if is_train:
            current_steps = logs.pop("step")
            total_steps = logs.pop("total_steps")
            epoch = logs.pop("epoch")
            total_epoch = logs.pop("total_epoch")
            num_steps_per_epoch = int(current_steps / epoch + 0.5)
            elapsed_time = logs.pop("elapsed_time")
            remaining_time = logs.pop("remaining_time")
            
            logs_item.extend([
                f'cur_step: [{current_steps-int(epoch)*num_steps_per_epoch}/{num_steps_per_epoch}]',
                f'epoch: [{epoch:.1f}/{total_epoch}]',
                f'total_steps: [{current_steps}/{total_steps}]'
            ])
        
        for name, val in logs.items():
            if isinstance(val, float):
                if "learning_rate" in name:
                    val = f'{val:.6f}'
                else:
                    val = f'{val:.4f}'
            if val is not None:
                logs_item.append(f'{name}: {val}')

        if is_train:
            logs_item.extend([
                f'elapsed_time: {self.get_time_tags(elapsed_time)}',
                f'remaining_time: {self.get_time_tags(remaining_time)}',
            ])
        
        log_str += ', '.join(logs_item)

        return log_str
