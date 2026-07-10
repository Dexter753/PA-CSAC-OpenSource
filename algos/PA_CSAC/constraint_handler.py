import numpy as np
import torch


class ConstraintHandler:
    def get_multiplier(self) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, qc_norm: torch.Tensor, cost_limit_norm: float) -> torch.Tensor:
        raise NotImplementedError

    def step(self):
        pass

    def zero_grad(self):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict):
        pass

    def save_to_ckpt(self, ckpt: dict):
        pass

    @staticmethod
    def create(method: str, penalty_weight: float = 1.0, device: str = "cpu",
               lambda_lr: float = 5e-4, lambda_max: float = 30.0):
        if method == "lagrangian":
            return LagrangianHandler(device=device, lr=lambda_lr, lambda_max=lambda_max)
        if method == "penalty":
            return PenaltyHandler(penalty_weight=penalty_weight, device=device)
        raise ValueError(f"Unknown constraint_method: {method}")


class PenaltyHandler(ConstraintHandler):
    def __init__(self, penalty_weight: float = 1.0, device: str = "cpu"):
        self._device = torch.device(device)
        self.penalty_weight = float(penalty_weight)
        self._weight = torch.tensor(self.penalty_weight, device=self._device)

    def get_multiplier(self) -> torch.Tensor:
        return self._weight

    def compute_loss(self, qc_norm: torch.Tensor, cost_limit_norm: float) -> torch.Tensor:
        return torch.tensor(0.0, device=self._device)

    def state_dict(self) -> dict:
        return {"penalty_weight": self.penalty_weight}

    def load_state_dict(self, state: dict):
        self.penalty_weight = float(state.get("penalty_weight", self.penalty_weight))
        self._weight = torch.tensor(self.penalty_weight, device=self._device)


class LagrangianHandler(ConstraintHandler):
    def __init__(self, device: str = "cpu", lr: float = 5e-4, lambda_max: float = 30.0):
        self._device = torch.device(device)
        self.lambda_max = float(lambda_max)
        self.log_lambda = torch.tensor(
            np.log(0.5), dtype=torch.float32, requires_grad=True, device=self._device
        )
        self.opt = torch.optim.Adam([self.log_lambda], lr=float(lr))

    def get_multiplier(self) -> torch.Tensor:
        return torch.exp(self.log_lambda).clamp(1e-4, self.lambda_max)

    def compute_loss(self, qc_norm: torch.Tensor, cost_limit_norm: float) -> torch.Tensor:
        return -(self.log_lambda * (qc_norm.detach().mean() - cost_limit_norm))

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

    def state_dict(self) -> dict:
        return {"log_lambda": self.log_lambda.detach().clone(),
                "lambda_max": self.lambda_max}

    def load_state_dict(self, state: dict):
        if "log_lambda" in state:
            self.log_lambda.data = state["log_lambda"].to(self._device)
        if "lambda_max" in state:
            self.lambda_max = float(state["lambda_max"])

    def save_to_ckpt(self, ckpt: dict):
        ckpt["log_lambda"] = self.log_lambda.detach().clone()
