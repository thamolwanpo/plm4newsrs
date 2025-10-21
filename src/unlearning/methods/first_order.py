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

    Key Formula:
    θ_new = θ* - τ · H^(-1) · (∇ℓ(Z̃_forget, θ*) - ∇ℓ(Z_all, θ*) · |Z̃|/|Z|)

    First-order approximation (without Hessian inverse):
    θ_new = θ* - τ · (∇ℓ(Z̃_forget, θ*) - ∇ℓ(Z_retain, θ*))

    Where:
    - θ*: Current model parameters (trained on all data including forget set)
    - Z̃_forget: Forget set (data to unlearn)
    - Z_retain: Retain set (data to keep)
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
        3. Compute gradient difference: Δ∇ = ∇ℓ_forget - ∇ℓ_retain
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
        print(f"Formula: θ_new = θ* - α · (∇ℓ_forget - ∇ℓ_retain)")
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

        Implements: θ_new = θ* - α · (∇ℓ_forget - ∇ℓ_retain)

        Steps:
        1. Compute ∇ℓ(Z_forget, θ*) - average gradient on forget set
        2. Compute ∇ℓ(Z_retain, θ*) - average gradient on retain set
        3. Compute gradient difference with damping
        4. Update parameters in the direction of gradient difference

        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set

        Returns:
            Gradient norm (for monitoring)
        """
        # Step 1: Compute gradient on forget set
        print("  1/4 Computing ∇ℓ_forget...")
        grad_forget = self._compute_average_gradient(forget_loader)

        # Step 2: Compute gradient on retain set
        print("  2/4 Computing ∇ℓ_retain...")
        grad_retain = self._compute_average_gradient(retain_loader)

        # Step 3: Compute gradient difference
        print("  3/4 Computing gradient difference...")
        grad_diff = {}
        total_norm = 0.0

        for name in grad_forget.keys():
            if name in grad_retain:
                # Δ∇ = ∇ℓ_forget - ∇ℓ_retain
                diff = grad_forget[name] - grad_retain[name]

                # Apply damping for stability
                if self.damping > 0:
                    diff = diff / (1.0 + self.damping)

                grad_diff[name] = diff
                total_norm += torch.sum(diff**2).item()

        grad_norm = torch.sqrt(torch.tensor(total_norm)).item()

        # Step 4: Update parameters
        print("  4/4 Updating parameters...")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in grad_diff:
                    # θ_new = θ* - α · Δ∇
                    param.data = param.data - self.learning_rate * grad_diff[name]

        return grad_norm

    def _compute_average_gradient(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """
        Compute average gradient over entire dataset.

        This computes: (1/|Z|) · Σ ∇ℓ(z_i, θ)

        Args:
            dataloader: DataLoader to compute gradients on

        Returns:
            Dictionary mapping parameter names to average gradients
        """
        self.model.train()
        self.model.zero_grad()

        # Accumulate gradients
        accumulated_grads = {}
        num_samples = 0

        for batch in dataloader:
            if batch is None:
                continue

            # Move to device
            if "device_indicator" in batch:
                batch["device_indicator"] = batch["device_indicator"].to(self.device)

            labels = batch["label"].to(self.device)
            batch_size = labels.size(0)

            # Forward pass
            scores = self.model(batch)
            loss = self.criterion(scores, labels)

            # Backward pass
            loss.backward()

            # Accumulate gradients (weighted by batch size)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in accumulated_grads:
                        accumulated_grads[name] = param.grad.clone() * batch_size
                    else:
                        accumulated_grads[name] += param.grad.clone() * batch_size

            num_samples += batch_size

            # Zero gradients for next iteration
            self.model.zero_grad()

        # Compute average
        avg_grads = {}
        for name, grad_sum in accumulated_grads.items():
            avg_grads[name] = grad_sum / num_samples

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
