import argparse
import logging
import numpy as np

from hmap.trainer_evaluator.ResNet8CIFAR10 import ResNet8CIFAR10
from hmap.trainer_evaluator.ResNet8CIFAR100 import ResNet8CIFAR100

from hmap.trainer_evaluator.ResNet20CIFAR10 import ResNet20CIFAR10
from hmap.trainer_evaluator.ResNet20CIFAR100 import ResNet20CIFAR100

from hmap.trainer_evaluator.AlexNetCIFAR10 import AlexNetCIFAR10
from hmap.trainer_evaluator.AlexNetCIFAR100 import AlexNetCIFAR100

from hmap.trainer_evaluator.VGG16CIFAR10 import VGG16CIFAR10
from hmap.trainer_evaluator.VGG16CIFAR100 import VGG16CIFAR100





def all_analog_layer_ids(model) -> list[int]:
    model.convert_layers_to_analog([])
    ids = []
    for _, module in model.named_modules():
        if hasattr(module, "ind_analog_layer"):
            try:
                ids.append(int(module.ind_analog_layer))
            except Exception:
                continue
    return sorted(set(ids))

def eval_avg(te, batch_size: int, num_workers: int, reps: int, drift: bool, t_eval: float) -> tuple[float, float]:
    te.model.eval()
    scores = []
    for _ in range(reps):
        if drift:
            te.model.drift_analog_weights(t_inference=t_eval)
        scores.append(te.evaluate(batch_size=batch_size, num_workers=num_workers))
    return float(np.mean(scores)), float(np.std(scores))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="resnet.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--t_eval", type=float, default=86400.0)
    parser.add_argument("--fully_analog", action="store_true", help="Convert all mappable layers to analog before eval")
    args = parser.parse_args()

    te = AlexNetCIFAR100()
    te.set_model()

    analog_ids = []
    if args.fully_analog:
        analog_ids = all_analog_layer_ids(te.model)
        te.model.convert_layers_to_analog(analog_ids)
        logging.info(f"Converted to analog: {len(analog_ids)} layers")

    te.load_checkpoint(checkpoint_path=args.checkpoint)
    logging.info(f"Loaded checkpoint: {args.checkpoint}")

    

    mean_acc, std_acc = eval_avg(
        te,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        reps=args.reps,
        drift=args.drift,
        t_eval=args.t_eval,
    )
    print(f"Accuracy (avg over {args.reps}): mean {mean_acc:2.2f}, std {std_acc:2.2f}")

if __name__ == "__main__":
    main()