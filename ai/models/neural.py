"""
ai/models/neural.py - Neural network model wrappers.

RESPONSIBILITY:
Provide MLP support with sklearn or numpy, and torch-backed deep sequence
architectures when torch is installed.

VERSION: 1.0.0
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from ai.models.base import BaseModel, ModelTask, flatten_features, flatten_target


# ==============================================================================
# NUMPY MLP
# ==============================================================================


class _NumpyMLP:
    """Single-hidden-layer neural network trained with numpy."""

    def __init__(
        self,
        task: ModelTask,
        hidden_units: int = 64,
        learning_rate: float = 0.01,
        epochs: int = 200,
        reg_lambda: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.hidden_units = max(2, int(hidden_units))
        self.learning_rate = float(learning_rate)
        self.epochs = max(1, int(epochs))
        self.reg_lambda = float(reg_lambda)
        self.rng = np.random.default_rng(random_state)
        self.classes_: NDArray[np.generic] | None = None
        self.w1_: NDArray[np.floating] | None = None
        self.b1_: NDArray[np.floating] | None = None
        self.w2_: NDArray[np.floating] | None = None
        self.b2_: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> "_NumpyMLP":
        x = flatten_features(X)
        target = flatten_target(y)
        n_features = x.shape[1]
        if self.task == ModelTask.CLASSIFICATION:
            self.classes_, encoded = np.unique(target, return_inverse=True)
            output_dim = len(self.classes_)
            y_train = np.eye(output_dim)[encoded]
        else:
            output_dim = 1
            y_train = target.astype(float).reshape(-1, 1)

        scale = 1.0 / np.sqrt(max(1, n_features))
        self.w1_ = self.rng.normal(0.0, scale, size=(n_features, self.hidden_units))
        self.b1_ = np.zeros((1, self.hidden_units), dtype=float)
        self.w2_ = self.rng.normal(0.0, scale, size=(self.hidden_units, output_dim))
        self.b2_ = np.zeros((1, output_dim), dtype=float)

        for _ in range(self.epochs):
            hidden, output = self._forward(x)
            if self.task == ModelTask.CLASSIFICATION:
                delta2 = (output - y_train) / float(len(x))
            else:
                delta2 = 2.0 * (output - y_train) / float(len(x))
            grad_w2 = hidden.T @ delta2 + self.reg_lambda * self.w2_
            grad_b2 = np.sum(delta2, axis=0, keepdims=True)
            delta1 = (delta2 @ self.w2_.T) * (1.0 - hidden * hidden)
            grad_w1 = x.T @ delta1 + self.reg_lambda * self.w1_
            grad_b1 = np.sum(delta1, axis=0, keepdims=True)
            self.w2_ -= self.learning_rate * grad_w2
            self.b2_ -= self.learning_rate * grad_b2
            self.w1_ -= self.learning_rate * grad_w1
            self.b1_ -= self.learning_rate * grad_b1
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        x = flatten_features(X)
        _, output = self._forward(x)
        if self.task == ModelTask.CLASSIFICATION:
            if self.classes_ is None:
                raise RuntimeError("Numpy MLP has no fitted classes")
            return self.classes_[np.argmax(output, axis=1)]
        return output.reshape(-1)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.task != ModelTask.CLASSIFICATION:
            raise RuntimeError("Probabilities are only available for classification MLPs")
        x = flatten_features(X)
        _, output = self._forward(x)
        return output

    def _forward(self, X: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        if self.w1_ is None or self.b1_ is None or self.w2_ is None or self.b2_ is None:
            raise RuntimeError("Numpy MLP must be fitted before prediction")
        hidden = np.tanh(X @ self.w1_ + self.b1_)
        raw = hidden @ self.w2_ + self.b2_
        if self.task == ModelTask.CLASSIFICATION:
            raw = raw - np.max(raw, axis=1, keepdims=True)
            exp = np.exp(raw)
            denom = np.sum(exp, axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            return hidden, exp / denom
        return hidden, raw


class MLPModel(BaseModel):
    """MLP wrapper using sklearn when available and numpy otherwise."""

    estimator_: Any = None

    def _make_estimator(self) -> Any:
        hidden = int(self.params.get("hidden_units", self.config.model.lstm_units))
        epochs = int(self.params.get("max_iter", self.params.get("epochs", 200)))
        try:
            neural_network = import_module("sklearn.neural_network")
            cls_name = "MLPClassifier" if self.task == ModelTask.CLASSIFICATION else "MLPRegressor"
            estimator_cls = getattr(neural_network, cls_name)
            return estimator_cls(
                hidden_layer_sizes=tuple(self.params.get("hidden_layer_sizes", (hidden,))),
                learning_rate_init=float(self.params.get("learning_rate", self.config.model.learning_rate)),
                max_iter=epochs,
                random_state=self.config.model.random_state,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "sklearn" or str(exc.name).startswith("sklearn."):
                return _NumpyMLP(
                    task=ModelTask.from_value(self.task),
                    hidden_units=hidden,
                    learning_rate=float(self.params.get("learning_rate", self.config.model.learning_rate)),
                    epochs=epochs,
                    reg_lambda=float(self.params.get("reg_lambda", self.config.model.reg_lambda)),
                    random_state=self.config.model.random_state,
                )
            raise

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(flatten_features(X), flatten_target(y))
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.estimator_ is None:
            raise RuntimeError("MLPModel must be fitted before prediction")
        return np.asarray(self.estimator_.predict(flatten_features(X)))

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION:
            return None
        if self.estimator_ is None:
            raise RuntimeError("MLPModel must be fitted before probability prediction")
        if hasattr(self.estimator_, "predict_proba"):
            return np.asarray(self.estimator_.predict_proba(flatten_features(X)), dtype=float)
        return None


# ==============================================================================
# TORCH SEQUENCE MODELS
# ==============================================================================


def _require_torch() -> Any:
    """Import torch or raise a concise optional dependency error."""
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError("Deep neural sequence models require torch. Install torch to use this model.") from exc
        raise


class _TorchSequenceModel(BaseModel):
    """Torch-backed sequence classifier/regressor."""

    architecture: str = "lstm"
    estimator_: Any = None
    classes_: NDArray[np.generic] | None = None

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        torch = _require_torch()
        nn = torch.nn
        x_np = self._sequence_array(X)
        target = flatten_target(y)
        input_dim = int(x_np.shape[-1])
        output_dim = 1
        if self.task == ModelTask.CLASSIFICATION:
            self.classes_, encoded = np.unique(target, return_inverse=True)
            output_dim = len(self.classes_)
            y_tensor = torch.as_tensor(encoded, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.as_tensor(target.astype(float).reshape(-1, 1), dtype=torch.float32)
            loss_fn = nn.MSELoss()

        model = self._build_torch_model(torch, input_dim=input_dim, output_dim=output_dim)
        device = self._device(torch)
        model.to(device)
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32).to(device)
        y_tensor = y_tensor.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params.get("learning_rate", self.config.model.learning_rate)),
        )
        epochs = int(self.params.get("epochs", 20))
        for _ in range(max(1, epochs)):
            model.train()
            optimizer.zero_grad()
            out = model(x_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()
        self.estimator_ = model.cpu()
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        torch = _require_torch()
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction")
        self.estimator_.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(self._sequence_array(X), dtype=torch.float32)
            output = self.estimator_(x_tensor).detach().cpu().numpy()
        if self.task == ModelTask.CLASSIFICATION:
            if self.classes_ is None:
                raise RuntimeError("Torch sequence model has no fitted classes")
            return self.classes_[np.argmax(output, axis=1)]
        return output.reshape(-1)

    def predict_proba(self, X: NDArray[np.floating]) -> Optional[NDArray[np.floating]]:
        if self.task != ModelTask.CLASSIFICATION:
            return None
        torch = _require_torch()
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before probability prediction")
        self.estimator_.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(self._sequence_array(X), dtype=torch.float32)
            logits = self.estimator_(x_tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        return np.asarray(probs, dtype=float)

    def _sequence_array(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 2:
            return arr.reshape(arr.shape[0], 1, arr.shape[1])
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected 2D or 3D sequence input, got shape {arr.shape}")

    def _device(self, torch: Any) -> Any:
        requested = str(self.params.get("device", self.config.training.device)).lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_torch_model(self, torch: Any, input_dim: int, output_dim: int) -> Any:
        nn = torch.nn
        hidden = int(self.params.get("hidden_units", self.config.model.lstm_units))
        layers = int(self.params.get("layers", self.config.model.lstm_layers))
        dropout = float(self.params.get("dropout", self.config.model.dropout))
        architecture = self.architecture

        class SequenceNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if architecture == "gru":
                    self.core = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
                    self.mode = "recurrent"
                elif architecture == "transformer":
                    heads = int(max(1, min(hidden, 4)))
                    self.project = nn.Linear(input_dim, hidden)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=hidden,
                        nhead=heads,
                        dim_feedforward=hidden * 2,
                        dropout=dropout,
                        batch_first=True,
                    )
                    self.core = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                    self.mode = "transformer"
                elif architecture == "tcn":
                    self.core = nn.Sequential(
                        nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                        nn.ReLU(),
                    )
                    self.mode = "conv"
                elif architecture == "cnn_lstm":
                    self.conv = nn.Sequential(nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1), nn.ReLU())
                    self.core = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
                    self.mode = "cnn_lstm"
                else:
                    self.core = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
                    self.mode = "recurrent"
                self.head = nn.Linear(hidden, output_dim)

            def forward(self, x: Any) -> Any:
                if self.mode == "transformer":
                    encoded = self.core(self.project(x))
                    pooled = encoded[:, -1, :]
                elif self.mode == "conv":
                    encoded = self.core(x.transpose(1, 2))
                    pooled = encoded[:, :, -1]
                elif self.mode == "cnn_lstm":
                    conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
                    encoded, _ = self.core(conv)
                    pooled = encoded[:, -1, :]
                else:
                    encoded, _ = self.core(x)
                    pooled = encoded[:, -1, :]
                return self.head(pooled)

        return SequenceNet()


class LSTMModel(_TorchSequenceModel):
    """Torch LSTM sequence model."""

    architecture = "lstm"


class GRUModel(_TorchSequenceModel):
    """Torch GRU sequence model."""

    architecture = "gru"


class TransformerModel(_TorchSequenceModel):
    """Torch Transformer encoder sequence model."""

    architecture = "transformer"


class TCNModel(_TorchSequenceModel):
    """Torch temporal convolutional network."""

    architecture = "tcn"


class CNNLSTMModel(_TorchSequenceModel):
    """Torch CNN-LSTM hybrid sequence model."""

    architecture = "cnn_lstm"


class AutoEncoderModel(BaseModel):
    """Torch autoencoder returning reconstruction error predictions."""

    estimator_: Any = None

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        X_val: NDArray[np.floating] | None = None,
        y_val: NDArray[np.floating] | None = None,
    ) -> BaseModel:
        torch = _require_torch()
        nn = torch.nn
        x_np = flatten_features(X).astype(float)
        input_dim = int(x_np.shape[1])
        hidden = int(self.params.get("hidden_units", self.config.model.transformer_dim))

        class AutoEncoderNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2))
                self.decoder = nn.Sequential(nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))

            def forward(self, x: Any) -> Any:
                return self.decoder(self.encoder(x))

        model = AutoEncoderNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.params.get("learning_rate", 0.001)))
        loss_fn = nn.MSELoss()
        x_tensor = torch.as_tensor(x_np, dtype=torch.float32)
        for _ in range(max(1, int(self.params.get("epochs", 30)))):
            optimizer.zero_grad()
            recon = model(x_tensor)
            loss = loss_fn(recon, x_tensor)
            loss.backward()
            optimizer.step()
        self.estimator_ = model
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        torch = _require_torch()
        if self.estimator_ is None:
            raise RuntimeError("AutoEncoderModel must be fitted before prediction")
        x_np = flatten_features(X).astype(float)
        with torch.no_grad():
            x_tensor = torch.as_tensor(x_np, dtype=torch.float32)
            recon = self.estimator_(x_tensor).detach().cpu().numpy()
        return np.mean((x_np - recon) ** 2, axis=1)


NEURAL_MODELS: Dict[str, type[BaseModel]] = {
    "mlp": MLPModel,
    "neural_mlp": MLPModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TransformerModel,
    "tcn": TCNModel,
    "cnn_lstm": CNNLSTMModel,
    "cnn-lstm": CNNLSTMModel,
    "autoencoder": AutoEncoderModel,
    "auto_encoder": AutoEncoderModel,
}
