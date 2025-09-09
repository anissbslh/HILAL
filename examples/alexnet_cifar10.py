import torch
from hmap.trainer_evaluator.AlexNetCIFAR10 import AlexNetCIFAR10

#from hmap.methods.LH import LH, LHConfig
#from hmap.methods.LH_eli_only import LH, LHConfig
#from hmap.methods.LH_r_only import LH, LHConfig

from hmap.methods.LH_topeigen_only import LH, LHConfig
#from hmap.methods.LH_topeigen_only_div_params import LH, LHConfig
#from hmap.methods.LH_trace_only import LH, LHConfig

if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=AlexNetCIFAR10(),
        checkpoint_path="alexnet_c10.pt",
        train_batch_size=256,
        eval_batch_size=256,
        digital_lr=1e-2,
        digital_momentum=0.85,
        analog_lr=1e-3,
        analog_momentum=0.85,
        drop_threshold=0.0,        # kept for compatibility (unused)
        num_workers=8,
        num_steps=-1,              
        logging_freq=50,
        t_eval=86400.0,
        evaluation_reps=6,
        patience=5,                
        seed=42,

        # Hessian descriptor + clustering controls
        hutchinson_probes=30,
        lanczos_steps=30,
        kmeans_restarts=8,
        sensitivity_batches=8,
        gamma_weight=0.08,
    )
    lh = LH(config)
    lh.set_baseline_score()
    lh.run()