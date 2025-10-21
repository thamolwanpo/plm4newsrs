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

        print(f"First-Order Unlearning initialized")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Steps: {self.num_steps}")
        print(f"  Damping: {self.damping}")

    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
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
            grad_norm = self._apply_unlearning_step(forget_loader, retain_loader)

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
        self, forget_loader: torch.utils.data.DataLoader, retain_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Apply single unlearning update step.

        Based on original implementation:
        - Z = data to FORGET
        - Z̃ = data to RETAIN (corrected)

        Update: θ_new = θ* - τ · (∇ℓ(Z̃_retain) - ∇ℓ(Z_forget))

        This moves:
        - TOWARD minimizing loss on retain set
        - AWAY FROM minimizing loss on forget set
        """
        # Step 1: Compute gradient on forget set (Z)
        print("  1/4 Computing ∇ℓ(Z_forget)...")
        grad_forget = self._compute_average_gradient(forget_loader)

        # Step 2: Compute gradient on retain set (Z̃)
        print("  2/4 Computing ∇ℓ(Z̃_retain)...")
        grad_retain = self._compute_average_gradient(retain_loader)

        # Step 3: Compute gradient difference
        print("  3/4 Computing gradient difference...")
        grad_diff = {}
        total_norm = 0.0

        # Track gradient magnitudes for diagnostics
        forget_norm = 0.0
        retain_norm = 0.0
        skipped_frozen = 0
        skipped_missing = 0

        for name, param in self.model.named_parameters():
            # CRITICAL: Skip frozen parameters
            if not param.requires_grad:
                skipped_frozen += 1
                continue

            # Skip if gradients not available for this parameter
            if name not in grad_forget or name not in grad_retain:
                skipped_missing += 1
                continue

            # Accumulate norms
            forget_norm += torch.sum(grad_forget[name] ** 2).item()
            retain_norm += torch.sum(grad_retain[name] ** 2).item()

            # CRITICAL FIX: Match original implementation
            # Δ = ∇ℓ(Z̃_retain) - ∇ℓ(Z_forget)
            diff = grad_retain[name] - grad_forget[name]

            # Apply damping for numerical stability
            if self.damping > 0:
                diff = diff / (1.0 + self.damping)

            grad_diff[name] = diff
            total_norm += torch.sum(diff**2).item()

        grad_norm = torch.sqrt(torch.tensor(total_norm)).item()

        # Diagnostic output
        print(f"     ∇ℓ(Z_forget) norm: {torch.sqrt(torch.tensor(forget_norm)).item():.6f}")
        print(f"     ∇ℓ(Z̃_retain) norm: {torch.sqrt(torch.tensor(retain_norm)).item():.6f}")
        print(f"     Difference norm: {grad_norm:.6f}")
        print(f"     Step size: {self.learning_rate * grad_norm:.6f}")
        if skipped_frozen > 0:
            print(f"     Skipped {skipped_frozen} frozen parameters")
        if skipped_missing > 0:
            print(f"     Skipped {skipped_missing} parameters (missing gradients)")

        # Step 4: Update parameters
        # θ_new = θ* - τ · (∇ℓ_retain - ∇ℓ_forget)
        print("  4/4 Updating parameters...")
        num_params_updated = 0
        update_stats = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Only update if we have a gradient difference computed
                if name in grad_diff:
                    # Apply update
                    param.data = param.data - self.learning_rate * grad_diff[name]
                    num_params_updated += param.numel()

                    # Track statistics (for debugging)
                    update_norm = (self.learning_rate * grad_diff[name]).norm().item()
                    param_norm = param.data.norm().item()
                    update_stats.append(
                        {
                            "param": name,
                            "update_norm": update_norm,
                            "param_norm": param_norm,
                            "relative_change": update_norm / (param_norm + 1e-8),
                        }
                    )

        print(f"     Updated {num_params_updated:,} trainable parameters")

        # Print top 3 largest updates (for debugging)
        if update_stats:
            update_stats.sort(key=lambda x: x["relative_change"], reverse=True)
            print(f"     Largest relative changes:")
            for stat in update_stats[:3]:
                # Truncate long parameter names
                param_name = stat["param"]
                if len(param_name) > 50:
                    param_name = param_name[:47] + "..."
                print(f"       {param_name}: {stat['relative_change']:.6f}")

        return grad_norm

    def _compute_average_gradient(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute average gradient over entire dataset.
        Only computes gradients for trainable parameters (requires_grad=True).

        This computes: (1/|Z|) · Σ ∇ℓ(z_i, θ)

        Args:
            dataloader: DataLoader to compute gradients on

        Returns:
            Dictionary mapping parameter names to average gradients
            (only includes trainable parameters)
        """
        self.model.train()
        self.model.zero_grad()

        # Accumulate gradients (only for trainable params)
        accumulated_grads = {}
        num_batches = 0  # Count batches, not samples

        for batch in dataloader:
            if batch is None:
                continue

            # Move to device
            if "device_indicator" in batch:
                batch["device_indicator"] = batch["device_indicator"].to(self.device)

            labels = batch["label"].to(self.device)

            # Forward pass
            scores = self.model(batch)
            loss = self.criterion(scores, labels)

            # Backward pass
            loss.backward()

            # Accumulate gradients (already averaged within batch by CrossEntropyLoss)
            # ONLY for trainable parameters
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue  # Skip frozen parameters

                if param.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = param.grad.clone()
                    else:
                        accumulated_grads[name] += param.grad.clone()

            num_batches += 1

            # Zero gradients for next iteration
            self.model.zero_grad()

        # Compute average over batches
        avg_grads = {}
        for name, grad_sum in accumulated_grads.items():
            avg_grads[name] = grad_sum / num_batches

        # Count trainable vs frozen
        trainable_count = len(avg_grads)
        total_count = sum(1 for _ in self.model.named_parameters())
        frozen_count = total_count - trainable_count

        if frozen_count > 0:
            print(
                f"     Computed gradients for {trainable_count} trainable params ({frozen_count} frozen)"
            )

        return avg_grads

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return method-specific hyperparameters."""
        return {
            "method": "first_order",
            "algorithm": "Machine Unlearning of Features and Labels",
            "formula": "θ_new = θ* - α · (∇ℓ_forget - ∇ℓ_retain)",
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
            "damping": self.damping,
        }
