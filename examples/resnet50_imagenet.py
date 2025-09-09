import hmap
import torch
from hmap.trainer_evaluator.ResNet50ImageNet import ResNet50ImageNet
from hmap.methods.LH import LH, LHConfig

if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=ResNet50ImageNet(),
        checkpoint_path="resnet50_imagenet.pt",
        train_batch_size=128,
        eval_batch_size=256,
        digital_lr=1e-3,
        digital_momentum=0.85,
        analog_lr=1e-4,
        analog_momentum=0.85,
        drop_threshold=0.0,        # kept for compatibility (unused)
        num_workers=8,
        num_steps=-1,              # auto: one pass over the train loader
        logging_freq=50,
        t_eval=86400.0,
        evaluation_reps=5,
        patience=5,
        seed=42,

        # Hessian descriptor + clustering controls
        hutchinson_probes=20,
        lanczos_steps=20,
        kmeans_restarts=8,
        sensitivity_batches=5,
        gamma_weight=0.08,
    )
    lh = LH(config)
    lh.set_baseline_score()
    lh.run()