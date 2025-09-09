import logging
import torch
from aihwkit.optim import AnalogSGD

from .TrainerEvaluator import TrainerEvaluator
from lionheart.datasets import ImageNet
from lionheart.models import ViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ViTImageNet(TrainerEvaluator):
    def instantiate_model(self):
        # Pretrained on ImageNet-1k
        model = ViT()
        return model.to(device)

    def instantiate_dataset(self):
        return ImageNet()

    def instantiate_optimizer(self, digital_lr: float, digital_momentum: float, analog_lr: float, analog_momentum: float):
        digital_parameters, analog_parameters = self.digital_analog_parameters(self.model)
        return AnalogSGD(
            [
                {
                    "params": analog_parameters,
                    "lr": analog_lr,
                    "momentum": analog_momentum,
                },
                {
                    "params": digital_parameters,
                    "lr": digital_lr,
                    "momentum": digital_momentum,
                },
            ],
        )

    def instantiate_scheduler(self):
        assert self.optimizer is not None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def train(self, num_steps: int, batch_size: int, num_workers: int, logging_freq: int = 50):
        assert self.model is not None
        assert self.optimizer is not None

        train_loader = self.dataset.load_train_data(batch_size=batch_size, num_workers=num_workers, validation=False)
        self.model = self.model.train().to(device)
        criterion = torch.nn.CrossEntropyLoss()

        current_step = 0
        while current_step < num_steps:
            for _, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = self.model(images)
                loss = criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.model.zero_grad(set_to_none=True)

                if current_step == 0 or (current_step + 1) % logging_freq == 0:
                    logging.info("current_step: %d,\tloss: %2.4f" % (current_step, loss.item()))

                current_step += 1
                if current_step >= num_steps:
                    # Keep the same return pattern as your MobileBERT trainer
                    return max(loss.item(), 1e-12) / max(current_step, 1)

    def evaluate(self, batch_size: int, num_workers: int):
        assert self.model is not None
        self.model = self.model.eval().to(device)

        # Use the official ImageNet val split
        val_loader = self.dataset.load_test_data(batch_size=batch_size, shuffle=False, num_workers=num_workers)

        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = self.model(images)  # logits [N, 1000]
                total += targets.size(0)

                # Top-1
                _, pred1 = outputs.max(1)
                top1_correct += pred1.eq(targets).sum().item()

                # Top-5
                _, pred5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                top5_correct += pred5.eq(targets.view(-1, 1)).any(dim=1).sum().item()

        top1_acc = 100.0 * top1_correct / max(total, 1)
        top5_acc = 100.0 * top5_correct / max(total, 1)
        logging.info("ImageNet Val - Top1: %2.2f%%, Top5: %2.2f%%" % (top1_acc, top5_acc))

        # Return Top-1 to mirror a single-number score API
        return top1_acc