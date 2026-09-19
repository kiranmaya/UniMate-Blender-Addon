"""Run UniMate inference with EMA weights stored as SafeTensors."""

import os
import sys

sys.path.insert(0, os.getcwd())

import tyro
from safetensors.torch import load_model

import unimate.inference.sample as sample


_original_loader = sample._load_checkpoint


def _load_checkpoint(model, model_path, config):
    if os.path.splitext(model_path)[1].lower() == ".safetensors":
        load_model(model, model_path, strict=True)
        sample.logger.info(f"Loaded SafeTensors weights: {model_path}")
        return
    _original_loader(model, model_path, config)


sample._load_checkpoint = _load_checkpoint


if __name__ == "__main__":
    sample.main(tyro.cli(sample.InferenceArgs))
