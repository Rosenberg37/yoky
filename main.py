import argparse
import logging
import os
from pathlib import Path

import yoky
from yoky import utils
from yoky.models import EntityDetection
from yoky.utils import data

logger = logging.getLogger("yoky")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process advance parameter.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Whether start in debug environment of cuda.')
    parser.add_argument('--cache_in_cur', default=False, action='store_true',
                        help='Whether cache files in the current directory.')
    environment_args = parser.parse_known_args()[0]

    if environment_args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if environment_args.cache_in_cur:
        yoky.cache_root = Path("cache")

    corpus_args = data.CorpusArguments().parse()
    model_args = EntityDetection.ModelArguments().parse()
    train_args = utils.TrainArguments().parse()

    corpus_name = corpus_args.corpus_name
    file_handler = logging.FileHandler(f"result/{corpus_name}/record.log")
    file_handler.setFormatter(yoky.log_format)
    logger.addHandler(file_handler)
    logger.info(
        "Main procedure started with:\n"
        f"{model_args}\n"
        f"{corpus_args}\n"
        f"{train_args}\n"
    )

    corpus = utils.Corpus(f"data/{corpus_name}", corpus_args)
    model_args.add('metadata', corpus.metadata)
    model = EntityDetection(model_args)

    trainer = utils.YokyTrainer(model, corpus, train_args)
    trainer()
