import math
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb

import lib.callbacks as callbacks
from lib.loggers import WandbLogger
from lib.arg_utils import define_args

from lib import NotALightningTrainer
from lib import nomenclature
from lib.forge import VersionCommand

from transformers import get_linear_schedule_with_warmup

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

VersionCommand().run()

args = define_args(
    extra_args = [
        ('--output_dir', {'default': 'test', 'type': str, 'required': False}),
    ]
)

wandb.init(project = 'nli-stress-test', group = args.group, entity='cosmadrian')
wandb.config.update(vars(args))

train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)

architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

evaluators = [
    nomenclature.EVALUATORS[evaluator_args.name](args, architecture, evaluator_args.args)
    for evaluator_args in args.evaluators
]

wandb_logger = WandbLogger()

checkpoint_callback_best = callbacks.ModelCheckpoint(
    args = args,
    name = ' 🔥 Best Checkpoint Overall 🔥',
    monitor = args.model_checkpoint['monitor_quantity'],
    dirpath = f'checkpoints/{args.group}:{args.name}/best/',
    save_best_only = True,
    direction = args.model_checkpoint['direction'],
    filename=f'epoch={{epoch}}-{args.model_checkpoint["monitor_quantity"]}={{{args.model_checkpoint["monitor_quantity"]}:.4f}}',
)

total_steps = math.ceil(args.epochs * len(train_dataloader))
warmup_steps = int(total_steps * args.warmup_percent)

scheduler = get_linear_schedule_with_warmup(model.configure_optimizers(lr = 2e-5 / 4), num_warmup_steps=warmup_steps, num_training_steps = total_steps)

lr_callback = callbacks.LambdaCallback(
    on_batch_end = lambda: scheduler.step()
)

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: wandb_logger.log('lr', scheduler.get_last_lr()[0])
)

if args.debug:
    print("[🐞DEBUG MODE🐞] Removing ModelCheckpoint ... ")
    callbacks = [lr_callback, lr_logger]
else:
    callbacks = [
        checkpoint_callback_best,
        lr_callback,
        lr_logger,
    ]

trainer = NotALightningTrainer(
    args = args,
    callbacks = callbacks,
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)
