import os
import uuid
import logging
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt

from aihwkit.nn import AnalogConv2d, AnalogLinear
from hmap.trainer_evaluator import TrainerEvaluator
from hmap.models.common.Linear import Linear as LHLinear
from hmap.models.common.Conv2d import Conv2d as LHConv2d
from .utils import rgetattr, seed_everything

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
@dataclass
class LHConfig:
    # Required
    trainer_evaluator: TrainerEvaluator
    checkpoint_path: str
    train_batch_size: int
    eval_batch_size: int
    digital_lr: float
    digital_momentum: float
    analog_lr: float
    analog_momentum: float

    # Legacy fields kept for compatibility (not used for decisions)
    drop_threshold: float

    # Runtimes
    num_workers: int = 4
    num_steps: int = -1            # -1 => one full pass over train loader
    patience: int = 1              # kept for compatibility (unused)
    t_eval: float = 86400
    evaluation_reps: int = 10
    logging_freq: int = 200
    seed: int = None
    early_stop_patience: int = 5   # kept for compatibility (unused now)
    retrain_epochs: int = 100     # single retraining phase length

    # Descriptor + clustering controls
    hutchinson_probes: int = 10
    lanczos_steps: int = 20
    kmeans_restarts: int = 8
    sensitivity_batches: int = 1
    gamma_weight: float = 0.08


class LH:
    def __init__(self, config: LHConfig):
        self.config = config
        self.baseline_score = -1.0

        # Mapping bookkeeping
        self.ind_analog_layers: List[int] = []
        self.reverse_id_lookup: Dict[int, str] = {}   # id -> dotted module path
        self.id_lookup: Dict[str, int] = {}           # dotted module path -> id
        

        # Descriptor storage
        self.layer_ids: List[int] = []                # ordered list of ids
        self.descriptors: Dict[int, Dict[str, float]] = {}  # id -> {"eli_norm": float, "r": float, "trace": float, "lambda_max": float}
        self.mac_ratios: Dict[int, float] = {}        # id -> mac_ratio

        # Clustering results
        self.robust_ids_ordered: List[int] = []       # robust ids sorted by eli_norm asc
        self.sensitive_ids: List[int] = []

        seed_everything(self.config.seed)

    # -------------------------
    # Public API
    # -------------------------
    def set_baseline_score(self):
        te = self.config.trainer_evaluator
        te.set_model()
        te.load_checkpoint(self.config.checkpoint_path)
        te.model.convert_layers_to_digital()
        self.baseline_score = te.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        logging.info("Set baseline score to: %s" % self.baseline_score)

    def run(self):
        # Start timing execution
        import time
        import json
        from datetime import datetime
        start_time = time.time()
        
        # Store all results in this dict
        results = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checkpoint_path": self.config.checkpoint_path,
                "uuid": str(uuid.uuid4()),
                "config": {k: str(v) if k == "trainer_evaluator" else v for k, v in self.config.__dict__.items()},
            },
            "descriptors": {},
            "mapping": {
                "robust_layers": [],
                "sensitive_layers": [],
                "mapped_layers": []
            },
            "performance": {
                "baseline": 0.0,
                "final_no_drift": {"mean": 0.0, "std": 0.0},
                "final_with_drift": {"mean": 0.0, "std": 0.0}
            },
            "resources": {
                "digital_macs": 0,
                "analog_macs": 0,
                "analog_ratio": 0.0
            },
            "timing": {
                "total_seconds": 0
            }
        }

        if self.baseline_score < 0:
            self.set_baseline_score()
        
        # Store baseline score
        results["performance"]["baseline"] = float(self.baseline_score)

        te = self.config.trainer_evaluator
        te.set_model(eval=False)
        te.load_checkpoint(self.config.checkpoint_path)
        te.model.convert_layers_to_analog([])

        # Map names <-> ids
        self._index_wrapped_layers(te.model)

        # Compute per-layer MAC ratios first (used in descriptors)
        self._compute_mac_ratios()

        # Compute descriptors (ELI_norm, r) and attach mac_ratio
        self._compute_descriptors()
        
        # Store all layer descriptors
        for layer_id, descriptor in self.descriptors.items():
            layer_name = self.reverse_id_lookup.get(layer_id, f"unknown-{layer_id}")
            results["descriptors"][str(layer_id)] = {
                "name": layer_name,
                **descriptor
            }

        # Cluster layers into robust vs sensitive using k=2 KMeans
        self._cluster_layers()
        
        # Store clustering results
        results["mapping"]["robust_layers"] = [int(i) for i in self.robust_ids_ordered]
        results["mapping"]["sensitive_layers"] = [int(i) for i in self.sensitive_ids]

        # Prepare mapping: map ONE robust layer at a time (progressive mapping)
        self.uuid = results["metadata"]["uuid"]
        self.updated_checkpoint_path = os.path.splitext(self.config.checkpoint_path)[0] + f"_LH_{self.uuid}.pt"

        # Start with no layers mapped to analog
        self.ind_analog_layers = []
        logging.info(f"Starting progressive mapping with {len(self.robust_ids_ordered)} robust layers.")

        # Save initial digital checkpoint
        te.model.convert_layers_to_analog([])
        te.save_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=[])

        # Compute steps/epoch if requested
        if self.config.num_steps == -1:
            self.config.num_steps = len(
                te.dataset.load_train_data(batch_size=self.config.train_batch_size, num_workers=self.config.num_workers, validation=False)
            )

        # Progressive mapping - map one layer at a time
        patience_window = 5  # Convergence patience window
        for layer_idx, layer_id in enumerate(self.robust_ids_ordered):
            layer_name = self.reverse_id_lookup.get(layer_id, f"unknown-{layer_id}")
            logging.info(f"Mapping layer [{layer_idx+1}/{len(self.robust_ids_ordered)}] {layer_id} ({layer_name})")
            
            # Add this layer to the mapped layers
            self.ind_analog_layers.append(layer_id)
            
            # Load best checkpoint so far and convert the current set of layers
            te.load_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)
            te.model.convert_layers_to_analog(self.ind_analog_layers)
            
            # Set optimizer/scheduler for this phase
            te.set_optimizer(
                digital_lr=self.config.digital_lr,
                digital_momentum=self.config.digital_momentum,
                analog_lr=self.config.analog_lr,
                analog_momentum=self.config.analog_momentum,
            )
            te.set_scheduler()

            # Retrain with patience window
            best_score = -float("inf")
            epochs_without_improvement = 0
            
            for epoch_idx in range(1, self.config.retrain_epochs + 1):
                te.train(
                    num_steps=self.config.num_steps,
                    batch_size=self.config.train_batch_size,
                    num_workers=self.config.num_workers,
                    logging_freq=self.config.logging_freq,
                )

                # Evaluate (optionally with drift and repetitions)
                scores = []
                te.model.eval()
                for _ in range(self.config.evaluation_reps):
                    scores.append(
                        te.evaluate(
                            batch_size=self.config.eval_batch_size,
                            num_workers=self.config.num_workers,
                        )
                    )
                avg_score = float(np.mean(scores))

                if avg_score > best_score + 1e-6:
                    best_score = avg_score
                    te.save_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)
                    logging.info(f"[Layer {layer_id}] Epoch {epoch_idx}: improved to {avg_score:2.2f}; checkpoint saved.")
                    epochs_without_improvement = 0  # Reset counter
                else:
                    epochs_without_improvement += 1
                    logging.info(f"[Layer {layer_id}] Epoch {epoch_idx}: {avg_score:2.2f} (no improvement for {epochs_without_improvement} epochs).")
                
                # Check for convergence - early stopping with patience window
                if epochs_without_improvement >= patience_window:
                    logging.info(f"[Layer {layer_id}] Early stopping after {epoch_idx} epochs (no improvement for {patience_window} epochs)")
                    break

            # Load best checkpoint for this layer before moving to next layer
            te.load_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)

        # Store final mapped layers
        results["mapping"]["mapped_layers"] = [int(i) for i in self.ind_analog_layers]

        # Final evaluation and MAC ratio
        te.load_checkpoint(checkpoint_path=self.updated_checkpoint_path)
        logging.info("Checkpoint path: %s" % self.updated_checkpoint_path)
        digital_macs, analog_macs = self._det_analog_digital_mac_ratio(self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)
        logging.info(f'analog macs : {analog_macs}')
        logging.info(f'digital macs : {digital_macs}')
        ratio_analog = float(analog_macs) / (float(float(digital_macs) + float(analog_macs)))
        logging.info(f'ratio_analog : {ratio_analog}')
        
        # Store MAC counts
        results["resources"]["digital_macs"] = float(digital_macs)
        results["resources"]["analog_macs"] = float(analog_macs)
        results["resources"]["analog_ratio"] = float(ratio_analog)

        # Print model summary
        print(te.model)

        # Evaluate without drift
        final_scores = []
        te.model.eval()
        for _ in range(self.config.evaluation_reps):
            final_scores.append(
                te.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
            )
        final_scores = np.array(final_scores)
        logging.info("Final score: mean %2.2f,\tstd: %2.2f" % (final_scores.mean(), final_scores.std()))
        
        # Store results without drift
        results["performance"]["final_no_drift"]["mean"] = float(final_scores.mean())
        results["performance"]["final_no_drift"]["std"] = float(final_scores.std())

        # Evaluate with drift
        final_scores = []
        te.model.eval()
        for _ in range(self.config.evaluation_reps):
            te.model.drift_analog_weights(t_inference=self.config.t_eval)
            final_scores.append(
                te.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
            )
        final_scores = np.array(final_scores)
        logging.info("[1-d] Final score: mean %2.2f,\tstd: %2.2f" % (final_scores.mean(), final_scores.std()))
        
        # Store results with drift
        results["performance"]["final_with_drift"]["mean"] = float(final_scores.mean())
        results["performance"]["final_with_drift"]["std"] = float(final_scores.std())
        
        # Compute and store total execution time
        end_time = time.time()
        total_time = end_time - start_time
        results["timing"]["total_seconds"] = float(total_time)
        logging.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Save results to JSON file
        results_path = os.path.splitext(self.updated_checkpoint_path)[0] + "ELI_ONLY_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to: {results_path}")
        
        return results
    



    # -------------------------
    # Internal: descriptor computation and clustering
    # -------------------------
    def _index_wrapped_layers(self, model: nn.Module):
        """Populate reverse_id_lookup and id_lookup from wrapped Linear/Conv2d modules."""
        self.reverse_id_lookup.clear()
        self.id_lookup.clear()
        self.layer_ids.clear()

        for name, module in model.named_modules():
            if isinstance(module, (LHLinear, LHConv2d)):
                if hasattr(module, "ind_analog_layer"):
                    idx = int(module.ind_analog_layer)
                    self.reverse_id_lookup[idx] = name
                    self.id_lookup[name] = idx
                    self.layer_ids.append(idx)

        # determinism
        self.layer_ids = sorted(self.layer_ids)

        logging.info(f"Indexed {len(self.layer_ids)} mappable layers.")

    def _compute_descriptors(self):
        """Compute ELI_norm and r=lambda_max/trace per mappable layer."""
        te = self.config.trainer_evaluator
        model = te.model

        # small train loader for sensitivity statistics
        train_loader = te.dataset.load_train_data(
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            validation=False,
        )
        data_iter = iter(train_loader)

        # we will reuse up to sensitivity_batches mini-batches
        batches = []
        for _ in range(self.config.sensitivity_batches):
            try:
                batches.append(next(data_iter))
            except StopIteration:
                break
        if len(batches) == 0:
            raise RuntimeError("No batches available from train loader to compute descriptors.")

        self.descriptors.clear()

        # Iterate wrapped layers (Conv2d/Linear wrappers)
        for layer_id in self.layer_ids:
            name = self.reverse_id_lookup[layer_id]
            wrapper = rgetattr(model, name)
            assert isinstance(wrapper, (LHLinear, LHConv2d))
            params = []
            # Use underlying op's parameters
            if hasattr(wrapper, "layer"):
                for p in wrapper.layer.parameters():
                    if p.requires_grad:
                        params.append(p)
            if len(params) == 0:
                # Nothing to optimize; skip
                self.descriptors[layer_id] = {"eli_norm": 0.0, "r": 0.0, "trace": 0.0, "lambda_max": 0.0}
                continue

            # Compute per-batch trace estimate (Hutchinson) and top eigen (power iteration)
            traces = []
            lambdas = []

            # Get sigma^2 := (gamma * max|W|)^2 (weight tensor only)
            max_abs_w = 0.0
            if hasattr(wrapper.layer, "weight") and wrapper.layer.weight is not None:
                max_abs_w = float(wrapper.layer.weight.detach().abs().max().item())
            else:
                # Fallback: consider all params
                for p in params:
                    if p.numel() > 0:
                        max_abs_w = max(max_abs_w, float(p.detach().abs().max().item()))
            sigma2 = (self.config.gamma_weight * max_abs_w) ** 2

            # number of parameters in this layer (weights + bias if any)
            n_params = int(sum(int(p.numel()) for p in params))

            for batch in batches:
                # Build a fresh loss graph per layer to control memory
                loss = self._compute_loss(model, batch)

                # First-order grads w.r.t. layer params
                grads1 = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
                grads1 = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads1, params)]

                # Hutchinson trace estimate
                vHv_sum = 0.0
                for _ in range(self.config.hutchinson_probes):
                    v_list = [self._rademacher_like(p) for p in params]
                    g_dot_v = sum((g * v).sum() for g, v in zip(grads1, v_list))
                    hv_list = torch.autograd.grad(g_dot_v, params, retain_graph=True, allow_unused=True)
                    hv_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv_list, params)]
                    vHv = sum((v * h).sum() for v, h in zip(v_list, hv_list))
                    vHv_sum += float(vHv.detach().item())
                trace_est = vHv_sum / float(max(1, self.config.hutchinson_probes))
                traces.append(trace_est)

                # Power iteration to approximate top eigenvalue
                lam_est = self._power_iteration_top_eig(params, grads1, self.config.lanczos_steps)
                lambdas.append(lam_est)

                # Cleanup per-batch graph
                del grads1, loss
                for p in params:
                    if p.grad is not None:
                        p.grad = None
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            trace = float(np.mean(traces)) if len(traces) else 0.0
            lam_max = float(np.mean(lambdas)) if len(lambdas) else 0.0

            eli_norm = 0.0
            r_ratio = 0.0
            if n_params > 0 and trace > 0.0:
                eli_norm = 0.5 * sigma2 * (trace / float(n_params))
                r_ratio = lam_max / trace

            mac_ratio = float(self.mac_ratios.get(layer_id, 0.0))

            self.descriptors[layer_id] = {
                "eli_norm": float(eli_norm),
                "r":      float(r_ratio),
                "mac_ratio": mac_ratio,
                "trace":  float(trace),
                "lambda_max": float(lam_max),
            }
            logging.info(
                f"Descriptor [{layer_id}] {name}: "
                f"ELI_norm={eli_norm:.6e}, r={r_ratio:.6e}, mac_ratio={mac_ratio:.4f}, "
                f"trace={trace:.6e}, lambda_max={lam_max:.6e}"
            )

    def _cluster_layers(self):
        """Run k-means (k=2) over 1D descriptors [ELI_norm] to split sensitive vs robust."""
        if len(self.descriptors) == 0:
            raise RuntimeError("Descriptors not computed.")

        ids = sorted(self.descriptors.keys())
        # Use only ELI_norm for clustering
        X = np.array(
            [[self.descriptors[i]["eli_norm"]] for i in ids],
            dtype=np.float64
        )

        # Standardize features across layers (z-score)
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-12
        Xz = (X - X_mean) / X_std

        labels, centers = self._kmeans_2d(Xz, k=2, n_init=self.config.kmeans_restarts, max_iter=100)

        # Favor robust cluster with low ELI (smaller ELI_norm means more robust)
        # The cluster with the smaller ELI_norm centroid is the robust one
        robust_cluster = int(np.argmin(centers[:, 0]))
        sensitive_cluster = 1 - robust_cluster

        self.sensitive_ids = [ids[i] for i, c in enumerate(labels) if int(c) == sensitive_cluster]
        robust_ids = [ids[i] for i, c in enumerate(labels) if int(c) == robust_cluster]

        # Order robust: prefer low ELI_norm only
        self.robust_ids_ordered = sorted(
            robust_ids,
            key=lambda lid: self.descriptors[lid]["eli_norm"]
        )

        logging.info(f"Sensitive ids: {self.sensitive_ids}")
        logging.info(f"Robust ids (ordered): {self.robust_ids_ordered}")

        # Save 2D clustering plot (ELI_norm vs layer index)
        try:
            base = os.path.splitext(self.config.checkpoint_path)[0]
            out_path = f"{base}_cluster_ELI.png"

            eli_values = np.array([self.descriptors[i]["eli_norm"] for i in ids], dtype=np.float64)
            layer_indices = np.arange(len(ids))
            
            # Use fixed marker size
            marker_size = 100

            mask_robust = np.isin(ids, self.robust_ids_ordered)
            mask_sensitive = np.isin(ids, self.sensitive_ids)

            plt.figure(figsize=(8, 4))
            # Use a bar plot to better visualize 1D data
            plt.bar(layer_indices[mask_robust], eli_values[mask_robust], 
                    color="tab:blue", label="robust", alpha=0.7, width=0.6)
            plt.bar(layer_indices[mask_sensitive], eli_values[mask_sensitive], 
                    color="tab:red", label="sensitive", alpha=0.7, width=0.6)
            
            # Add layer IDs as labels
            for i, lid in enumerate(ids):
                plt.text(i, eli_values[i] * 1.05, str(lid), ha='center', fontsize=8)

            plt.xlabel("Layer Index")
            plt.ylabel("ELI_norm")
            plt.title("Layer clustering based on ELI_norm only")
            plt.grid(True, alpha=0.3, axis='y')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            logging.info(f"Saved clustering plot to: {out_path}")
        except Exception as e:
            logging.warning(f"Could not save clustering plot: {e}")
    # -------------------------
    # Internal: utilities
    # -------------------------
    def _compute_loss(self, model: nn.Module, batch) -> torch.Tensor:
        """Compute a scalar training loss for either CIFAR-10 or SQuAD batches."""
        # CIFAR-10: (data, target)
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]):
            data, target = batch
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            # Some wrapped layers return through hooks; outputs should be tensor
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, target)
            return loss

        # SQuAD (MobileBERT): (input_ids, attention_mask, token_type_ids, start_positions, end_positions)
        if isinstance(batch, (list, tuple)) and len(batch) >= 5:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            outputs = model(**inputs)
            # transformers models return loss as first element
            if hasattr(outputs, "loss"):
                return outputs.loss
            return outputs[0]

        # Fallback: try to call model and take first element as loss
        outputs = model(batch)
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            return outputs[0]
        raise RuntimeError("Unsupported batch format for loss computation.")

    def _rademacher_like(self, p: torch.Tensor) -> torch.Tensor:
        """Generate Rademacher vector (+1/-1) like p."""
        return torch.randint_like(p, low=0, high=2, device=p.device, dtype=torch.int8).float().mul_(2.0).add_(-1.0)

    def _flatten_list(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors]) if len(tensors) else torch.tensor([], device=device)

    def _power_iteration_top_eig(self, params: List[torch.Tensor], grads1: List[torch.Tensor], steps: int) -> float:
        """Approximate top eigenvalue using power iteration with hvp from grads1."""
        # Initialize v
        v_list = [torch.randn_like(p) for p in params]
        v_flat = self._flatten_list([v for v in v_list])
        v_norm = torch.linalg.norm(v_flat) + 1e-12
        v_list = [v / v_norm for v in v_list]

        lam = 0.0
        for _ in range(max(1, steps)):
            # hv = H v via second grad of g.v
            g_dot_v = sum((g * v).sum() for g, v in zip(grads1, v_list))
            hv_list = torch.autograd.grad(g_dot_v, params, retain_graph=True, allow_unused=True)
            hv_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv_list, params)]

            v_flat = self._flatten_list(v_list)
            hv_flat = self._flatten_list(hv_list)

            denom = float(torch.dot(v_flat, v_flat).detach().item()) + 1e-12
            num = float(torch.dot(v_flat, hv_flat).detach().item())
            lam = num / denom

            # Normalize for next step
            hv_norm = torch.linalg.norm(hv_flat) + 1e-12
            v_list = [h / hv_norm for h in hv_list]

        return float(lam)

    def _kmeans_2d(self, X: np.ndarray, k: int = 2, n_init: int = 8, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simple KMeans for feature arrays (N x D); returns labels, centers."""
        assert X.ndim == 2 and X.shape[0] >= k
        best_inertia = float("inf")
        best_labels = None
        best_centers = None

        rng = np.random.default_rng(self.config.seed if self.config.seed is not None else None)
        N = X.shape[0]

        for _ in range(max(1, n_init)):
            # Random unique init indices
            init_idx = rng.choice(N, size=k, replace=False)
            centers = X[init_idx].copy()

            for _ in range(max_iter):
                # Assign
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)

                # Update
                new_centers = np.vstack([X[labels == c].mean(axis=0) if np.any(labels == c) else centers[c] for c in range(k)])
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift < 1e-6:
                    break

            inertia = 0.0
            for c in range(k):
                if np.any(labels == c):
                    diffs = X[labels == c] - centers[c]
                    inertia += float(np.sum(diffs * diffs))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

        return best_labels, best_centers

    def _det_analog_digital_mac_ratio(self, checkpoint_path: str, ind_analog_layers: List[int]) -> Dict[str, float]:
        te = self.config.trainer_evaluator
        assert checkpoint_path is not None
        te.set_model()
        te.load_checkpoint(self.config.checkpoint_path, ind_analog_layers=ind_analog_layers)
        te.model.convert_layers_to_analog(ind_analog_layers)
        te.model.set_track_MACs()
        te.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        MAC_ops = te.model.get_MACs()
        analog_MACs = 0.0
        digital_MACs = 0.0
        for name, ops in MAC_ops.items():
            layer = rgetattr(te.model, name).layer
            if isinstance(layer, (AnalogConv2d, AnalogLinear)):
                analog_MACs += float(ops)
            elif isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                digital_MACs += float(ops)
            else:
                raise NotImplementedError

        te.model.unset_track_MACs()
        return digital_MACs, analog_MACs

    def _compute_mac_ratios(self):
        """Compute per-layer MAC ratio = layer_MACs / total_MACs using the model's MAC tracker."""
        te = self.config.trainer_evaluator
        model = te.model

        # Ensure all-digital for counting (does not affect MACs but keeps it consistent)
        model.convert_layers_to_analog([])

        model.set_track_MACs()
        # Run one evaluation pass to populate MACs (uses validation/test data inside te.evaluate)
        _ = te.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        MAC_ops = model.get_MACs()
        model.unset_track_MACs()

        # Sum only over tracked modules
        total_macs = float(sum(float(v) for v in MAC_ops.values())) + 1e-12

        self.mac_ratios.clear()
        # Map module names to ids; only keep those that are mappable
        for name, ops in MAC_ops.items():
            if name in self.id_lookup:
                lid = self.id_lookup[name]
                self.mac_ratios[lid] = float(ops) / total_macs

        # Default 0.0 for any layer missing in the dict
        for lid in self.layer_ids:
            self.mac_ratios.setdefault(lid, 0.0)

        logging.info(f"Computed per-layer MAC ratios for {len(self.mac_ratios)} layers (total MACs={total_macs:.3e}).")