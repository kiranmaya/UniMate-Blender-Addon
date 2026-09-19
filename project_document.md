# UniMate Blender Add-on — Project Document

## Overview

UniMate Blender Add-on is a Blender 5.2 sidebar tool for generating
text-guided character motion with UniMate-compatible model weights. It runs
inference in an external Python environment, then imports the generated motion
into Blender as a selectable Action on the target armature.

Repository: <https://github.com/kiranmaya/UniMate-Blender-Addon>  
Release: <https://github.com/kiranmaya/UniMate-Blender-Addon/releases/tag/v0.1.0>

## Current status

- Public GitHub repository published.
- Installable Blender add-on release published.
- Blender 5.2 LTS tested.
- NVIDIA RTX 3060 Ti 8 GB tested.
- SafeTensors inference path verified.
- Demo character, conditioning data, screenshot, and configuration included.

## User workflow

1. Install the Blender add-on.
2. Install the official UniMate code in a separate Python 3.10 environment.
3. Download the compatible SafeTensors checkpoint from Hugging Face.
4. Set the UniMate root, Python executable, experiment, model, and conditioning paths in Blender.
5. Enter a prompt and action name.
6. Generate motion.
7. Choose a generated Action, apply it to the armature, and play it.

The Blender UI provides three separate selectors: a scene-aware Character
armature dropdown, a motion Action preset dropdown, and a UniMate Skeleton
dropdown populated from the configured `cond.npy`. This prevents Blender object
names from being mistaken for UniMate conditioning keys.

Generated runs retain the prompt, seed, source-motion path, log, and timing.

## Architecture

```text
Blender sidebar add-on
        |
        | launches external Python process
        v
UniMate inference environment (Python 3.10 + CUDA PyTorch)
        |
        | loads SafeTensors + skeleton conditioning
        v
Generated motion .npy
        |
        | decoded by Blender add-on
        v
Blender Action on the selected armature
```

The add-on intentionally keeps PyTorch outside Blender's bundled Python. This
avoids dependency conflicts and lets CUDA inference use the user's normal
UniMate environment.

## Components

| Path | Purpose |
|---|---|
| `unimate_blender/__init__.py` | Blender UI, subprocess generation, motion import, action controls |
| `unimate_blender/unimate_safetensors_runner.py` | Loads UniMate-compatible `.safetensors` weights directly |
| `config/littleKrishna_config.json` | Example experiment configuration |
| `example_assets/littleKrishna/` | Demo FBX, canonical FBX, conditioning, and rest clip |
| `docs/unimate-panel.jpg` | Interface reference screenshot |
| `README.md` | Installation, requirements, troubleshooting, and licensing guidance |

## Models and licensing

The official UniMate repository has released training and inference code, but
its README still states that official pretrained checkpoints are pending.

This project uses the independently trained third-party checkpoint at:

<https://huggingface.co/tarn59/UniMate-Weights>

It is not an official UniMate checkpoint. The add-on, repository, and README
must continue to make this distinction clear. The plugin source is MIT
licensed; model weights, UniMate, datasets, T5, and the demo character asset
retain their own licenses and usage terms.

## Tested performance

Test system:

| Item | Result |
|---|---|
| GPU | NVIDIA RTX 3060 Ti, 8 GB VRAM |
| Clip length | 60 frames at 30 FPS |
| Warm generation time | Approximately 26–40 seconds per clip |
| SafeTensors smoke-test time | 63 seconds for one clip; 84 seconds including process setup |
| Observed total GPU memory | Approximately 3.1 GB during inference |

First-run time is higher because the Flan-T5 text encoder must be downloaded
and initialized.

## Technical constraints

- The bundled example configuration supports up to 61 padded joints.
- The prepared `littleKrishna` example uses a 22-joint Mixamo-style core.
- Arbitrary FBX files require UniMate preprocessing to produce matching
  `cond.npy`, rest motion, and a compatible experiment configuration.
- Finger/helper bones may need pruning or weight merging.
- Motion quality depends on skeleton topology, bone names, conditioning,
  prompt, seed, GPU precision, and the independent checkpoint.

## Planned improvements

- Add a guided character-preprocessing operator.
- Add a built-in setup validator for paths, CUDA, model files, and conditioning keys.
- Add batch prompt generation.
- Add an NLA-track workflow for sequencing generated clips.
- Add Linux-specific installation screenshots and validation.
- Add motion preview renders and basic quality checks.

## Release checklist

- Keep model weights out of GitHub releases.
- Keep the unofficial-checkpoint disclaimer visible in every release.
- Preserve the separate demo-asset notice.
- Re-test SafeTensors loading after UniMate or Blender upgrades.
- Update system requirements with each supported GPU/Blender version.
