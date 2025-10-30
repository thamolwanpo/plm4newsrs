import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from src.unlearning.base import BaseUnlearningMethod
from src.unlearning.methods import register_method


@register_method("gradient_ascent")
class GradientAscentUnlearning(BaseUnlearningMethod):
    """
    Knowledge Unlearning via Gradient Ascent.

    Based on: "Knowledge Unlearning for Mitigating Privacy Risks in Language Models"

    The algorithm maximizes the loss on the forget set to make the model
    forget the target knowledge:

    L_UL(f_θ, x) = -Σ log(p_θ(x_t | x_<t))

    which is equivalent to performing gradient ascent:
    θ_new = θ + α · ∇ℓ(Z_forget, θ)

    Where:
    - θ: Current model parameters
    - Z_forget: Forget set (data to unlearn)
    - α: Learning rate (step size)
    - ∇ℓ: Gradient of loss on forget set

    The intuition: By maximizing loss on forget set, we push the model
    away from being able to predict/remember that data.

    Reference:
        "Knowledge Unlearning for Mitigating Privacy Risks in Language Models"
    """

    def __init__(self, model, model_config, unlearn_config, device):
        """Initialize gradient ascent unlearning."""
        super().__init__(model, model_config, unlearn_config, device)

        self.learning_rate = unlearn_config.learning_rate
        self.num_steps = unlearn_config.num_steps
        self.gradient_clip_val = unlearn_config.gradient_clip_val

        self.criterion = nn.CrossEntropyLoss()

        print(f"Gradient Ascent Unlearning initialized")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Steps: {self.num_steps}")
        print(f"  Gradient clipping: {self.gradient_clip_val}")

    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        validation_loader: Optional[torch.utils.data.DataLoader] = None,
        corrected_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> nn.Module:
        """
        Apply gradient ascent unlearning algorithm.

        Algorithm:
        1. For each step:
            a. Compute gradient on forget set: ∇ℓ(Z_forget)
            b. Update parameters in ASCENT direction: θ = θ + α · ∇ℓ
            c. Clip gradients for stability

        Args:
            forget_loader: DataLoader for forget set (data to unlearn)
            retain_loader: DataLoader for retain set (for evaluation)
            validation_loader: Optional validation data
            corrected_loader: Ignored (not needed for gradient ascent)

        Returns:
            Unlearned model
        """
        print(f"\n{'='*70}")
        print("GRADIENT ASCENT UNLEARNING")
        print(f"{'='*70}")
        print(f"Algorithm: Knowledge Unlearning via Loss Maximization")
        print(f"Formula: θ_new = θ + α · ∇ℓ(Z_forget, θ)")
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

        # Apply gradient ascent for num_steps
        for step in range(self.num_steps):
            print(f"{'─'*70}")
            print(f"GRADIENT ASCENT STEP {step + 1}/{self.num_steps}")
            print(f"{'─'*70}")

            # Apply single gradient ascent update
            avg_loss, grad_norm = self._apply_gradient_ascent_step(forget_loader)

            print(f"  Forget set loss: {avg_loss:.4f}")
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
        print("GRADIENT ASCENT UNLEARNING COMPLETE")
        print(f"{'='*70}\n")

        return self.model

    def _apply_gradient_ascent_step(self, forget_loader):
        """
        Apply gradient ascent by reversing the ranking.
        """
        self.model.train()

        # Store initial params for comparison
        initial_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()

        total_loss = 0.0
        total_grad_norm_raw = 0.0
        num_batches = 0

        # Track predictions BEFORE updates
        predictions_before = []
        with torch.no_grad():
            for batch in forget_loader:
                if batch is None:
                    continue
                batch_tensors = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                scores = self.model(batch_tensors)
                preds = torch.argmax(scores, dim=1)
                predictions_before.extend(preds.cpu().tolist())

        # Now do gradient ascent
        for batch in forget_loader:
            if batch is None:
                continue

            # Move tensors to device
            batch_tensors = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_tensors[k] = v.to(self.device)
                else:
                    batch_tensors[k] = v

            labels = batch_tensors["label"]

            # Zero gradients
            self.model.zero_grad()

            # Forward pass
            scores = self.model(batch_tensors)

            # Loss
            loss = self.criterion(scores, labels)

            # Backward pass
            loss.backward()

            # Calculate raw gradient norm
            raw_grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    raw_grad_norm += torch.sum(param.grad.data**2).item()

            total_grad_norm_raw += raw_grad_norm

            # GRADIENT ASCENT: θ = θ + α · ∇ℓ
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.data.add_(self.learning_rate * param.grad.data)

            total_loss += loss.item()
            num_batches += 1

            # Cleanup
            del batch_tensors, scores, labels, loss
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Track predictions AFTER updates
        predictions_after = []
        with torch.no_grad():
            for batch in forget_loader:
                if batch is None:
                    continue
                batch_tensors = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                scores = self.model(batch_tensors)
                preds = torch.argmax(scores, dim=1)
                predictions_after.extend(preds.cpu().tolist())

        # Calculate parameter changes
        max_param_change = 0.0
        total_param_change = 0.0
        num_params_changed = 0

        for name, param in self.model.named_parameters():
            if name in initial_params:
                change = torch.norm(param.data - initial_params[name]).item()
                max_param_change = max(max_param_change, change)
                total_param_change += change
                num_params_changed += 1

        avg_param_change = (
            total_param_change / num_params_changed if num_params_changed > 0 else 0.0
        )

        # Print diagnostics
        print(f"     Training loss: {total_loss / num_batches:.4f}")
        print(
            f"     Raw gradient norm: {torch.sqrt(torch.tensor(total_grad_norm_raw / num_batches)).item():.6f}"
        )
        print(f"     Learning rate: {self.learning_rate}")
        print(f"     Max param change: {max_param_change:.6f}")
        print(f"     Avg param change: {avg_param_change:.6f}")

        # Prediction changes
        pred_before_0 = sum(1 for p in predictions_before if p == 0) / len(predictions_before)
        pred_after_0 = sum(1 for p in predictions_after if p == 0) / len(predictions_after)
        print(f"     % predicting pos 0 BEFORE: {pred_before_0*100:.2f}%")
        print(f"     % predicting pos 0 AFTER: {pred_after_0*100:.2f}%")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = (
            torch.sqrt(torch.tensor(total_grad_norm_raw / num_batches)).item()
            if num_batches > 0
            else 0.0
        )

        return avg_loss, avg_grad_norm

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return method-specific hyperparameters."""
        return {
            "method": "gradient_ascent",
            "algorithm": "Knowledge Unlearning via Gradient Ascent",
            "formula": "θ_new = θ + α · ∇ℓ(Z_forget, θ)",
            "learning_rate": self.learning_rate,
            "num_steps": self.num_steps,
        }
