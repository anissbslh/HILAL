import os

import HILAL
import torch
from HILAL.trainer_evaluator.MobileBERTSquad import MobileBERTSquad
from HILAL.methods.HILAL import HILAL, HILALConfig


if __name__ == "__main__":
    config = HILALConfig(
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
    hilal = HILAL(config)
    hilal.set_baseline_score()
    hilal.run()