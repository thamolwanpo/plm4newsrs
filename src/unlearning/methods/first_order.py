# src/unlearning/methods/first_order.py

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import copy

from src.unlearning.base import BaseUnlearningMethod
from src.unlearning.methods import register_method


@register_method("first_order")
class FirstOrderUnlearning(BaseUnlearningMethod):
    """
    Machine Unlearning of Features and Labels (First-Order Approximation).

    Based on: "Machine Unlearning of Features and Labels"

    The algorithm approximates the optimal parameters after removing a subset
    of training data (forget set) without full retraining.

    First-order approximation (without Hessian inverse):
    θ_new = θ* - τ · (∇ℓ(Z̃_retain, θ*) - ∇ℓ(Z_forget, θ*))

    Where:
    - θ*: Current model parameters (trained on all data including forget set)
    - Z̃_retain: Retain set (data to keep)
    - Z_forget: Forget set (data to unlearn)
    - τ: Step size (learning rate)
    - ∇ℓ: Gradient of loss

    The intuition: We want to move parameters in the direction that increases
    loss on forget set while maintaining loss on retain set.

    Reference:
        Warnecke et al., "Machine Unlearning of Features and Labels", ICML 2023.
    """

    def __init__(self, model, model_config, unlearn_config, device):
        """Initialize first-order unlearning."""
        super().__init__(model, model_config, unlearn_config, device)

        # Unlearning parameters
        self.learning_rate = unlearn_config.learning_rate
        self.num_steps = unlearn_config.num_steps
        self.damping = unlearn_config.damping

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.label_correction_mode = getattr(unlearn_config, "use_label_correction", True)

        print(f"First-Order Unlearning initialized")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Steps: {self.num_steps}")
        print(f"  Damping: {self.damping}")
        print(f"  Mode: {'Label Correction' if self.label_correction_mode else 'Data Removal'}")

    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
        corrected_loader: Optional[torch.utils.data.DataLoader] = None,  # ADD THIS
    ) -> nn.Module:
        """
        Apply first-order unlearning algorithm.

        Algorithm:
        1. Compute average gradient on forget set: ∇ℓ_forget
        2. Compute average gradient on retain set: ∇ℓ_retain
        3. Compute gradient difference: Δ∇ = ∇ℓ_retain - ∇ℓ_forget
        4. Update parameters: θ_new = θ* - α · Δ∇

        Can be applied multiple times (num_steps) for stronger unlearning.

        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
            validation_loader: Optional validation data

        Returns:
            Unlearned model
        """
        print(f"\n{'='*70}")
        print("FIRST-ORDER UNLEARNING")
        print(f"{'='*70}")
        print(f"Algorithm: Machine Unlearning of Features and Labels")
        if corrected_loader is not None:
            print(f"Mode: Label Correction (paper's formula)")
            print(f"Formula: θ_new = θ* - α · (∇ℓ(Z̃_correct) - ∇ℓ(Z_wrong))")
        else:
            print(f"Mode: Data Removal (heuristic)")
            print(f"Formula: θ_new = θ* - α · (∇ℓ_retain - ∇ℓ_forget)")
        print(f"{'='*70}\n")

        # Evaluate before unlearning
        if validation_loader:
            print("=" * 50)
            print("BEFORE UNLEARNING")
            print("=" * 50)
            initial_forget = self.evaluate(forget_loader, self.criterion)
            initial_retain = self.evaluate(retain_loader, self.criterion)
            print(
                f"Forget set - Loss: {initial_forget['loss']:.4f}, Acc: {initial_forget['accuracy']:.4f}"
            )
            print(
                f"Retain set - Loss: {initial_retain['loss']:.4f}, Acc: {initial_retain['accuracy']:.4f}"
            )
            print()

        # Apply unlearning for num_steps
        for step in range(self.num_steps):
            print(f"{'─'*70}")
            print(f"UNLEARNING STEP {step + 1}/{self.num_steps}")
            print(f"{'─'*70}")

            # Apply single unlearning update
            grad_norm = self._apply_unlearning_step(forget_loader, retain_loader, corrected_loader)

            print(f"  Gradient norm: {grad_norm:.6f}")

            # Evaluate after this step
            if validation_loader and (
                step % max(1, self.num_steps // 5) == 0 or step == self.num_steps - 1
            ):
                forget_metrics = self.evaluate(forget_loader, self.criterion)
                retain_metrics = self.evaluate(retain_loader, self.criterion)

                print(
                    f"  Forget set - Loss: {forget_metrics['loss']:.4f}, Acc: {forget_metrics['accuracy']:.4f}"
                )
                print(
                    f"  Retain set - Loss: {retain_metrics['loss']:.4f}, Acc: {retain_metrics['accuracy']:.4f}"
                )

                self.log_progress(
                    step,
                    {
                        "forget_loss": forget_metrics["loss"],
                        "forget_acc": forget_metrics["accuracy"],
                        "retain_loss": retain_metrics["loss"],
                        "retain_acc": retain_metrics["accuracy"],
                        "grad_norm": grad_norm,
                    },
                )

            print()

        # Final evaluation
        if validation_loader:
            print("=" * 50)
            print("AFTER UNLEARNING")
            print("=" * 50)
            final_forget = self.evaluate(forget_loader, self.criterion)
            final_retain = self.evaluate(retain_loader, self.criterion)
            print(
                f"Forget set - Loss: {final_forget['loss']:.4f}, Acc: {final_forget['accuracy']:.4f}"
            )
            print(
                f"Retain set - Loss: {final_retain['loss']:.4f}, Acc: {final_retain['accuracy']:.4f}"
            )

            print(f"\n{'─'*70}")
            print("CHANGES:")
            print(f"{'─'*70}")
            print(f"Forget set:")
            print(f"  Loss: {final_forget['loss'] - initial_forget['loss']:+.4f}")
            print(f"  Accuracy: {final_forget['accuracy'] - initial_forget['accuracy']:+.4f}")
            print(f"Retain set:")
            print(f"  Loss: {final_retain['loss'] - initial_retain['loss']:+.4f}")
            print(f"  Accuracy: {final_retain['accuracy'] - initial_retain['accuracy']:+.4f}")

        print(f"\n{'='*70}")
        print("FIRST-ORDER UNLEARNING COMPLETE")
        print(f"{'='*70}\n")

        return self.model

    def _apply_unlearning_step(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        corrected_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> float:
        if self.label_correction_mode and corrected_loader is not None:
            print("  Using LABEL CORRECTION mode (paper's formula)")

            print("  1/3 Computing ∇ℓ(Z) - wrong labels...")
            grad_Z = self._compute_average_gradient(forget_loader)

            print("  2/3 Computing ∇ℓ(Z̃) - correct labels...")
            grad_Z_tilde = self._compute_average_gradient(corrected_loader)

            print("  3/3 Computing Δ = [∇ℓ(Z̃) - ∇ℓ(Z)]...")
            grad_diff = {}
            total_norm = 0.0
            z_norm = 0.0
            z_tilde_norm = 0.0

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in grad_Z or name not in grad_Z_tilde:
                    continue

                z_norm += torch.sum(grad_Z[name] ** 2).item()
                z_tilde_norm += torch.sum(grad_Z_tilde[name] ** 2).item()

                diff = grad_Z_tilde[name] - grad_Z[name]

                if self.damping > 0:
                    diff = diff / (1.0 + self.damping)

                grad_diff[name] = diff
                total_norm += torch.sum(diff**2).item()

            grad_norm = torch.sqrt(torch.tensor(total_norm)).item()

            print(f"     ∇ℓ(Z) norm: {torch.sqrt(torch.tensor(z_norm)).item():.6f}")
            print(f"     ∇ℓ(Z̃) norm: {torch.sqrt(torch.tensor(z_tilde_norm)).item():.6f}")
            print(f"     Difference norm: {grad_norm:.6f}")

        else:
            print("  Using DATA REMOVAL mode (heuristic)")

            print("  1/3 Computing ∇ℓ(Z_forget)...")
            grad_forget = self._compute_average_gradient(forget_loader)

            print("  2/3 Computing ∇ℓ(Z_retain)...")
            grad_retain = self._compute_average_gradient(retain_loader)

            # === FIX IS HERE: REVERSE GRADIENT ORDER ===
            print("  3/3 Computing Δ = ∇ℓ_retain - ∇ℓ_forget...")
            grad_diff = {}
            total_norm = 0.0
            forget_norm = 0.0
            retain_norm = 0.0

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in grad_forget or name not in grad_retain:
                    continue

                forget_norm += torch.sum(grad_forget[name] ** 2).item()
                retain_norm += torch.sum(grad_retain[name] ** 2).item()

                # *** CHANGED: Retain (Good) minus Forget (Bad) ***
                diff = grad_retain[name] - grad_forget[name]

                if self.damping > 0:
                    diff = diff / (1.0 + self.damping)

                grad_diff[name] = diff
                total_norm += torch.sum(diff**2).item()

            grad_norm = torch.sqrt(torch.tensor(total_norm)).item()

            print(f"     ∇ℓ(Z_forget) norm: {torch.sqrt(torch.tensor(forget_norm)).item():.6f}")
            print(f"     ∇ℓ(Z_retain) norm: {torch.sqrt(torch.tensor(retain_norm)).item():.6f}")
            print(f"     Difference norm: {grad_norm:.6f}")

        # === START OF MODIFIED SECTION ===
        print(f"  Updating parameters...")
        num_params_updated = 0

        # Calculate the full update vector: delta = -tau * grad_diff
        # Use self.learning_rate as the tau/eta value
        delta_updates = {name: -self.learning_rate * diff for name, diff in grad_diff.items()}

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in delta_updates:
                    # Apply update: param.data = param.data + delta
                    # This matches the param.add_(delta) logic from the first code.
                    param.data.add_(delta_updates[name])
                    num_params_updated += param.numel()

        print(f"     Updated {num_params_updated:,} trainable parameters")

        return grad_norm

    def _compute_average_gradient(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the AVERAGE gradient over the entire dataset.
        Returns: ∇ℓ = (1/N) * Σ∇ℓ_i
        """
        self.model.train()
        self.model.zero_grad()

        accumulated_grads = {}
        num_samples = 0

        for batch in dataloader:
            if batch is None:
                continue

            # Move ALL tensors in batch to device
            batch_tensors = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_tensors[k] = v.to(self.device)
                else:
                    batch_tensors[k] = v

            labels = batch_tensors["label"]
            batch_size = labels.size(0)

            # Forward pass
            scores = self.model(batch_tensors)
            loss = self.criterion(scores, labels)

            # Backward pass
            loss.backward()

            # Accumulate gradients (weighted by batch size)
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.grad is not None:
                    # Accumulate: sum += grad * batch_size
                    batch_grad_sum = param.grad.clone() * batch_size

                    if name not in accumulated_grads:
                        accumulated_grads[name] = batch_grad_sum
                    else:
                        accumulated_grads[name] += batch_grad_sum

            num_samples += batch_size

            # Zero gradients for next iteration
            self.model.zero_grad()

            # Cleanup
            import gc

            del batch_tensors, scores, labels, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # === CRITICAL FIX: NORMALIZE BY TOTAL SAMPLES ===
        if num_samples == 0:
            return {}

        for name in accumulated_grads:
            accumulated_grads[name] = accumulated_grads[name] / num_samples

        # Count trainable vs frozen
        trainable_count = len(accumulated_grads)
        total_count = sum(1 for _ in self.model.named_parameters())
        frozen_count = total_count - trainable_count

        if frozen_count > 0:
            print(
                f"     Computed gradients for {trainable_count} trainable params ({frozen_count} frozen)"
            )
        print(f"     Averaged over {num_samples} samples")

        return accumulated_grads

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return method-specific hyperparameters."""
        return {
            "method": "first_order",
            "algorithm": "Machine Unlearning of Features and Labels",
            "formula": "θ_new = θ* - α · (∇ℓ_retain - ∇ℓ_forget)",
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "damping": self.damping,
        }
