import os

os.environ["HF_HOME"] = "/gpfs/u/scratch/MCMI/MCMIbssl/shared/hf_cache"

import hmap
import torch
from hmap.trainer_evaluator.MobileBERTSquad import MobileBERTSquad
from hmap.methods.LH import LH, LHConfig


if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=MobileBERTSquad(),
        checkpoint_path='cp.pt',
        train_batch_size=4,
        eval_batch_size=4,
        digital_lr=1e-2,
        digital_momentum=0.85,
        analog_lr=1e-3,
        analog_momentum=0.85,
        drop_threshold=0.0,        # kept for compatibility (unused)
        num_workers=4,
        num_steps=-1,              # auto: one pass over the train loader
        logging_freq=50,
        t_eval=86400.0,
        evaluation_reps=2,
        patience=2,                # kept for compatibility (unused)
        seed=42,

        # Hessian descriptor + clustering controls
        hutchinson_probes=50,
        lanczos_steps=50,
        kmeans_restarts=8,
        sensitivity_batches=1,
        gamma_weight=0.08,
    )
    lh = LH(config)
    lh.set_baseline_score()
    lh.run()