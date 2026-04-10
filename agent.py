from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


def _state_to_vector(state: np.ndarray) -> np.ndarray:
    return state.astype(np.float32).reshape(1, -1)


def _transform_coords(row: int, col: int, size: int, transform_id: int) -> Tuple[int, int]:
    if transform_id == 0:
        return row, col
    if transform_id == 1:
        return col, size - 1 - row
    if transform_id == 2:
        return size - 1 - row, size - 1 - col
    if transform_id == 3:
        return size - 1 - col, row
    if transform_id == 4:
        return size - 1 - row, col
    if transform_id == 5:
        return row, size - 1 - col
    if transform_id == 6:
        return col, row
    if transform_id == 7:
        return size - 1 - col, size - 1 - row
    raise ValueError(f"Unknown transform_id: {transform_id}")


def transform_board(board: np.ndarray, transform_id: int) -> np.ndarray:
    size = board.shape[0]
    transformed = np.zeros_like(board)
    for row in range(size):
        for col in range(size):
            new_row, new_col = _transform_coords(row, col, size, transform_id)
            transformed[new_row, new_col] = board[row, col]
    return transformed


def transform_action(action: int, size: int, transform_id: int) -> int:
    row, col = divmod(int(action), size)
    new_row, new_col = _transform_coords(row, col, size, transform_id)
    return new_row * size + new_col


def symmetry_variants(board: np.ndarray, action: Optional[int] = None) -> List[Tuple[np.ndarray, Optional[int]]]:
    size = board.shape[0]
    variants: List[Tuple[np.ndarray, Optional[int]]] = []
    for transform_id in range(8):
        transformed_board = transform_board(board, transform_id)
        transformed_action = None if action is None else transform_action(action, size, transform_id)
        variants.append((transformed_board, transformed_action))
    return variants


def canonical_board_key(board: np.ndarray) -> bytes:
    transformed = [transform_board(board, transform_id).astype(np.int8).tobytes() for transform_id in range(8)]
    return min(transformed)


def canonical_action(board: np.ndarray, action: int) -> Tuple[bytes, int]:
    variants = symmetry_variants(board, action)
    encoded: List[Tuple[bytes, int]] = []
    for transformed_board, transformed_action in variants:
        encoded.append((transformed_board.astype(np.int8).tobytes(), int(transformed_action or 0)))
    return min(encoded, key=lambda item: item[0])


class DenseNetwork:
    """A compact NumPy MLP used as a lightweight DQN approximator."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int], output_dim: int, seed: int = 42) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = list(hidden_layers)
        rng = np.random.default_rng(seed)

        layer_sizes = [self.input_dim, *self.hidden_layers, self.output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for left, right in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (left + right))
            self.weights.append(rng.uniform(-limit, limit, size=(left, right)).astype(np.float32))
            self.biases.append(np.zeros(right, dtype=np.float32))

    def clone(self) -> "DenseNetwork":
        clone = DenseNetwork(self.input_dim, self.hidden_layers, self.output_dim)
        clone.weights = [weight.copy() for weight in self.weights]
        clone.biases = [bias.copy() for bias in self.biases]
        return clone

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations = x
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weight + bias
            if layer_index < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
        return activations

    def train_batch(self, x: np.ndarray, target: np.ndarray, learning_rate: float) -> float:
        x = np.asarray(x, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        activations = [x]
        pre_activations = []
        current = x
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            pre_activations.append(current)
            if layer_index < len(self.weights) - 1:
                current = np.maximum(current, 0.0)
            activations.append(current)

        prediction = activations[-1]
        error = prediction - target
        loss = float(np.mean(error * error))

        gradient = (2.0 / x.shape[0]) * error
        weights_snapshot = [weight.copy() for weight in self.weights]
        grad_weights: List[np.ndarray] = [np.zeros_like(weight) for weight in self.weights]
        grad_biases: List[np.ndarray] = [np.zeros_like(bias) for bias in self.biases]

        for layer_index in reversed(range(len(self.weights))):
            activation_prev = activations[layer_index]
            grad_weights[layer_index] = activation_prev.T @ gradient
            grad_biases[layer_index] = gradient.sum(axis=0)

            if layer_index > 0:
                gradient = gradient @ weights_snapshot[layer_index].T
                gradient = gradient * (pre_activations[layer_index - 1] > 0.0)

        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= learning_rate * grad_weights[layer_index]
            self.biases[layer_index] -= learning_rate * grad_biases[layer_index]

        return loss


class QLearningAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table: Dict[bytes, np.ndarray] = {}
        self.total_updates = 0

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        key = state.astype(np.int8).tobytes()
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[key]

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = True) -> int:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return 0
        if explore and np.random.random() < self.epsilon:
            return int(np.random.choice(legal_moves))

        q_values = self._get_q_values(state)
        masked = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked[legal_moves] = q_values[legal_moves]
        return int(np.argmax(masked))

    def learn_transition(self, transition: Transition) -> float:
        total_loss = 0.0
        for transform_id in range(8):
            state_variant = transform_board(transition.state, transform_id)
            next_state_variant = transform_board(transition.next_state, transform_id)
            action_variant = transform_action(transition.action, self.board_size, transform_id)
            state_key = state_variant.astype(np.int8).tobytes()
            q_values = self.q_table.setdefault(state_key, np.zeros(self.action_size, dtype=np.float32))
            next_q_values = self._get_q_values(next_state_variant)

            if transition.done:
                target_value = transition.reward
            else:
                legal_next = np.flatnonzero(next_state_variant.reshape(-1) == 0)
                if legal_next.size == 0:
                    target_value = transition.reward
                else:
                    target_value = transition.reward + self.gamma * float(np.max(next_q_values[legal_next]))

            prediction = float(q_values[action_variant])
            q_values[action_variant] = prediction + self.learning_rate * (target_value - prediction)
            total_loss += abs(target_value - prediction)

        self.total_updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss / 8.0

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "q_learning",
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "total_updates": self.total_updates,
            "q_table": self.q_table,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        agent = cls(
            board_size=payload["board_size"],
            learning_rate=payload["learning_rate"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.total_updates = payload.get("total_updates", 0)
        agent.q_table = payload["q_table"]
        return agent


class DQNAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.998,
        memory_size: int = 50_000,
        batch_size: int = 64,
        hidden_layers: Sequence[int] = (256, 128),
        target_update_every: int = 250,
    ) -> None:
        self.board_size = int(board_size)
        self.action_size = self.board_size * self.board_size
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.memory: Deque[Transition] = deque(maxlen=int(memory_size))
        self.model = DenseNetwork(self.action_size, self.hidden_layers, self.action_size)
        self.target_model = self.model.clone()
        self.train_steps = 0
        self.total_updates = 0

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = True) -> int:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return 0
        if explore and np.random.random() < self.epsilon:
            return int(np.random.choice(legal_moves))

        q_values = self.model.predict(_state_to_vector(state))[0]
        masked = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked[legal_moves] = q_values[legal_moves]
        return int(np.argmax(masked))

    def remember(self, transition: Transition) -> None:
        for transform_id in range(8):
            state_variant = transform_board(transition.state, transform_id)
            next_state_variant = transform_board(transition.next_state, transform_id)
            action_variant = transform_action(transition.action, self.board_size, transform_id)
            self.memory.append(
                Transition(
                    state=state_variant.astype(np.float32),
                    action=int(action_variant),
                    reward=float(transition.reward),
                    next_state=next_state_variant.astype(np.float32),
                    done=bool(transition.done),
                )
            )

    def learn_from_transition(self, transition: Transition) -> float:
        self.remember(transition)
        return self.replay()

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return 0.0

        batch = [self.memory[index] for index in np.random.choice(len(self.memory), self.batch_size, replace=False)]
        states = np.vstack([_state_to_vector(item.state) for item in batch])
        next_states = np.vstack([_state_to_vector(item.next_state) for item in batch])
        predicted = self.model.predict(states)
        targets = predicted.copy()
        next_q_values = self.target_model.predict(next_states)

        for row_index, item in enumerate(batch):
            if item.done:
                target_value = item.reward
            else:
                legal_next = np.flatnonzero(item.next_state.reshape(-1) == 0)
                if legal_next.size == 0:
                    target_value = item.reward
                else:
                    target_value = item.reward + self.gamma * float(np.max(next_q_values[row_index, legal_next]))
            targets[row_index, item.action] = target_value

        loss = self.model.train_batch(states, targets, self.learning_rate)
        self.train_steps += 1
        self.total_updates += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_model = self.model.clone()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "dqn",
            "board_size": self.board_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "hidden_layers": self.hidden_layers,
            "target_update_every": self.target_update_every,
            "train_steps": self.train_steps,
            "total_updates": self.total_updates,
            "model_weights": [weight.copy() for weight in self.model.weights],
            "model_biases": [bias.copy() for bias in self.model.biases],
            "target_weights": [weight.copy() for weight in self.target_model.weights],
            "target_biases": [bias.copy() for bias in self.target_model.biases],
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        agent = cls(
            board_size=payload["board_size"],
            learning_rate=payload["learning_rate"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            batch_size=payload["batch_size"],
            hidden_layers=payload["hidden_layers"],
            target_update_every=payload["target_update_every"],
        )
        agent.train_steps = payload.get("train_steps", 0)
        agent.total_updates = payload.get("total_updates", 0)
        agent.model.weights = [weight.copy() for weight in payload["model_weights"]]
        agent.model.biases = [bias.copy() for bias in payload["model_biases"]]
        agent.target_model.weights = [weight.copy() for weight in payload["target_weights"]]
        agent.target_model.biases = [bias.copy() for bias in payload["target_biases"]]
        return agent


def create_agent(
    board_size: int,
    algorithm: str = "auto",
    learning_rate: float = 0.1,
    gamma: float = 0.9,
) -> object:
    algorithm = algorithm.lower().strip()
    if algorithm == "auto":
        algorithm = "q_learning" if board_size <= 5 else "dqn"
    if algorithm in {"q", "q_learning", "q-learning"}:
        return QLearningAgent(board_size=board_size, learning_rate=learning_rate, gamma=gamma)
    if algorithm == "dqn":
        return DQNAgent(board_size=board_size, learning_rate=learning_rate, gamma=gamma)
    raise ValueError(f"Unknown algorithm: {algorithm}")
