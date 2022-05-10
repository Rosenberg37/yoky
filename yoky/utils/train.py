import argparse
import datetime
import logging
import os
from typing import Optional

import torch
import transformers
from accelerate import Accelerator
from torch import optim
from tqdm import tqdm

from yoky import utils, nn
from yoky.utils import Corpus, Arguments
from yoky.utils.data import YokyDataLoader

logger = logging.getLogger('yoky')


class TrainArguments(Arguments):
    def parse(self):
        parser = argparse.ArgumentParser(description='Process procedure parameter.')
        parser.add_argument('--checkpoint_name', type=str, default='base', help='Extra name of checkpoint.')
        parser.add_argument('--evaluate', type=str, nargs='*', help='Which dataset to evaluate on.')
        parser.add_argument('--batch_size', type=int, default=4, help='Batch size per iteration.')
        parser.add_argument('--epochs', type=int, default=5, help='How many epochs to run.')
        parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
        parser.add_argument('--max_grad_norm', type=float, default=1e0, help='The maximum norm of the per-sample gradients.')
        parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay rate of optimizer.')
        parser.add_argument('--lr_warmup', type=float, default=0.1, help='Warmup rate for scheduler.')
        parser.add_argument('--load', default=False, action='store_true', help='Whether load the parameters of model.')
        parser.add_argument('--use_cpu', default=False, action='store_true', help='Whether use CPU to train model.')
        parser.add_argument('--not_save_model', default=False, action='store_true', help='Whether save the best parameters.')
        self.__dict__.update(vars(parser.parse_known_args()[0]))
        return self


class YokyTrainer:
    def __init__(self, model: nn.Module, corpus: Corpus, args: TrainArguments):
        super().__init__()
        self.model = model
        self.evaluate = args.evaluate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.not_save_model = args.not_save_model
        self.max_grad_norm = args.max_grad_norm
        self.corpus = corpus
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.lr_warmup = args.lr_warmup

        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()

        self.checkpoint_path = f"result/{corpus.corpus_name}/models/{args.checkpoint_name}.pth"
        self.best_f1 = 0
        self.analysist = utils.Analysist(corpus, model)

        self.accelerator = Accelerator(cpu=args.use_cpu)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if args.load:
            self.load_checkpoint(self.model, self.checkpoint_path, self.optimizer)

    def __call__(self):
        train_dataloader = self.accelerator.prepare(self.get_train_dataloader())

        self.model.train()
        for epoch in range(self.epochs):
            t = tqdm(train_dataloader, desc=f"Epoch {epoch}")
            losses, runtimes = list(), list()
            for data in t:
                self.optimizer.zero_grad()
                loss = self.model(data)
                self.accelerator.backward(loss)

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

                losses.append(loss.item())
                t.set_postfix({'loss': loss.item(), 'avg_loss': sum(losses) / len(losses)})

            runtimes.append(t.format_dict['elapsed'])
            logger.info(f"Epoch {epoch}:\n"
                        f"Average_loss:{sum(losses) / len(losses)}\n"
                        f"Average_runtime:{datetime.timedelta(seconds=sum(runtimes) / len(runtimes))}")

            for name in self.evaluate:
                statistic = self.analysist.evaluate(name)
                f1 = statistic['all']['f1']

                logger.info(f"{name}:\n{statistic}")

                if not self.not_save_model and \
                        (name == 'test' or (name == 'dev' and 'test' not in self.evaluate)) \
                        and f1 > self.best_f1:
                    self.best_f1 = f1
                    logger.info(f"Save models with best performance on {name}\n")
                    self.save_checkpoint(self.model, self.checkpoint_path, self.optimizer)

    def get_train_dataloader(self) -> YokyDataLoader:
        return YokyDataLoader(
            self.corpus.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def get_eval_dataloader(self) -> YokyDataLoader:
        return YokyDataLoader(
            self.corpus.dev,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    def get_test_dataloader(self) -> YokyDataLoader:
        return YokyDataLoader(
            self.corpus.dev,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    def create_optimizer(self):
        named_parameters = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params_group = [
            {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.weight_decay},
            {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0}
        ]
        return optim.AdamW(params_group, lr=self.learning_rate)

    def create_scheduler(self):
        total_steps = (len(self.corpus) // self.batch_size + 1) * self.epochs
        return transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.lr_warmup * total_steps,
            num_training_steps=total_steps
        )

    @staticmethod
    def save_checkpoint(model: nn.Module, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None):
        torch.save({
            'models': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path, _use_new_zipfile_serialization=False)

    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str, optimizer: Optional[optim.Optimizer] = None):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['models'], strict=False)
            if len(missing_keys) == 0 and len(unexpected_keys) == 0 and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                logger.warning(f'Missing_keys:{missing_keys}\n'
                               f'Unexpected_keys:{unexpected_keys}')
        else:
            raise RuntimeError("Checkpoint not exist!")
