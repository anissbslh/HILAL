import HILAL
import torch
import logging
import numpy as np
from HILAL.trainer_evaluator.ResNet8CIFAR10 import ResNet8CIFAR10
from HILAL.trainer_evaluator.ResNet8CIFAR100 import ResNet8CIFAR100

from HILAL.trainer_evaluator.ResNet20CIFAR10 import ResNet20CIFAR10
from HILAL.trainer_evaluator.ResNet20CIFAR100 import ResNet20CIFAR100

from HILAL.trainer_evaluator.AlexNetCIFAR10 import AlexNetCIFAR10
from HILAL.trainer_evaluator.AlexNetCIFAR100 import AlexNetCIFAR100

from HILAL.trainer_evaluator.VGG16CIFAR10 import VGG16CIFAR10
from HILAL.trainer_evaluator.VGG16CIFAR100 import VGG16CIFAR100

def all_analog_layer_ids(model) -> list[int]:
    # Assign stable indices without converting any to analog
    model.convert_layers_to_analog([])
    ids = []
    for _, module in model.named_modules():
        if hasattr(module, "ind_analog_layer"):
            try:
                ids.append(int(module.ind_analog_layer))
            except Exception:
                continue
    ids = sorted(set(ids))
    print("layers", len(ids))
    return ids

def eval_avg(te, batch_size: int, num_workers: int, reps: int, drift: bool, t_eval: float = 86400.0) -> float:
    te.model.eval()
    scores = []
    for _ in range(reps):
        if drift:
            te.model.drift_analog_weights(t_inference=t_eval)
        scores.append(te.evaluate(batch_size=batch_size, num_workers=num_workers))
    return float(np.mean(scores)), float(np.std(scores))

def main():
    train_batch_size = 128
    eval_batch_size = 256
    num_workers = 4
    epochs = 200
    evaluation_reps = 10
    t_eval = 86400.0
    patience = 7  # patience window for early stopping
    checkpoint_path = "vgg16_c100.pt"  # path to pretrained weights

    te = VGG16CIFAR100() #adapt this
    te.set_model()

    # Load pretrained digital weights
    te.load_checkpoint(checkpoint_path=checkpoint_path)
    logging.info(f"Loaded pretrained checkpoint: {checkpoint_path}")

    # Baseline evaluation (digital; no drift)
    base_mean, base_std = eval_avg(
        te, batch_size=eval_batch_size, num_workers=num_workers, reps=evaluation_reps, drift=False, t_eval=t_eval
    )
    logging.info(f"Baseline accuracy (avg over {evaluation_reps}): mean {base_mean:2.2f}, std {base_std:2.2f}")
    print(f"Baseline accuracy: mean {base_mean:2.2f}, std {base_std:2.2f}")

    # Fully convert to analog
    analog_ids = all_analog_layer_ids(te.model)
    te.model.convert_layers_to_analog(analog_ids)
    logging.info(f"Converted to analog: {len(analog_ids)} layers")

    # Optimizer and scheduler for hardware-aware training
    te.set_optimizer(
        digital_lr=1e-2,
        digital_momentum=0.85,
        analog_lr=1e-3,
        analog_momentum=0.85,
    )
    te.set_scheduler()

    # Steps per epoch
    num_steps = len(te.dataset.load_train_data(batch_size=train_batch_size, num_workers=num_workers, validation=False))
    te.model.train()
    best_score = -float("inf")
    epochs_without_improvement = 0  # counter for early stopping

    for epoch in range(1, epochs + 1):
        te.train(
            num_steps=num_steps,
            batch_size=train_batch_size,
            num_workers=num_workers,
            logging_freq=100,
        )

        # Epoch evaluation (analog; with drift and averaging)
        mean_score, std_score = eval_avg(
            te, batch_size=eval_batch_size, num_workers=num_workers, reps=evaluation_reps, drift=False, t_eval=t_eval
        )
        logging.info(f"Epoch: {epoch}, Score mean {mean_score:2.2f} (std {std_score:2.2f})")

        if mean_score > best_score + 1e-6:
            te.save_checkpoint(checkpoint_path="vgg16_c100_analog.pt", ind_analog_layers=analog_ids)
            logging.info(f"New best (avg) score: {mean_score:2.2f}")
            best_score = mean_score
            epochs_without_improvement = 0  # reset counter when there's improvement
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    # Final evaluation: reload best checkpoint, convert, and average with drift
    te.load_checkpoint(checkpoint_path="vgg16_c100_analog.pt", ind_analog_layers=analog_ids)
    te.model.convert_layers_to_analog(analog_ids)
    final_mean, final_std = eval_avg(
        te, batch_size=eval_batch_size, num_workers=num_workers, reps=evaluation_reps, drift=False, t_eval=t_eval
    )
    final_mean_d, final_std_d = eval_avg(
        te, batch_size=eval_batch_size, num_workers=num_workers, reps=evaluation_reps, drift=True, t_eval=t_eval
    )
    logging.info(f"Final best accuracy (avg over {evaluation_reps}): mean {final_mean:2.2f}, std {final_std:2.2f}")
    print(f"Final best accuracy: mean {final_mean:2.2f}, std {final_std:2.2f}")

    logging.info(f"[Drift 1-d] Final best accuracy (avg over {evaluation_reps}): mean {final_mean_d:2.2f}, std {final_std_d:2.2f}")
    print(f"Final best accuracy: mean {final_mean_d:2.2f}, std {final_std_d:2.2f}")

if __name__ == "__main__":
    main()