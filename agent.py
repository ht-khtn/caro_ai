from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAX_BOARD_SIZE = 15
MAX_ABS_Q = 500.0
MAX_ABS_ACTIVATION = 1_000.0
MAX_ABS_GRADIENT = 5.0
MAX_ABS_WEIGHT = 12.0


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    board_size: int = MAX_BOARD_SIZE


def _state_to_vector(state: np.ndarray, board_size: int) -> np.ndarray:
    return _embed_board(state, board_size).astype(np.float32).reshape(1, -1)


def _safe_clip_scalar(value: float, limit: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -limit, limit))


def _board_offset(board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> int:
    return max(0, (canvas_size - int(board_size)) // 2)


def _embed_board(board: np.ndarray, board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> np.ndarray:
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.int8)
    offset = _board_offset(board_size, canvas_size)
    canvas[offset : offset + board_size, offset : offset + board_size] = board.astype(np.int8)
    return canvas


def _board_action_to_canvas(action: int, board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> int:
    offset = _board_offset(board_size, canvas_size)
    row, col = divmod(int(action), int(board_size))
    return (row + offset) * canvas_size + (col + offset)


def _canvas_action_to_board(action: int, board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> int:
    offset = _board_offset(board_size, canvas_size)
    row, col = divmod(int(action), canvas_size)
    return (row - offset) * board_size + (col - offset)


def _legal_canvas_actions(legal_moves: Sequence[int], board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> List[int]:
    return [_board_action_to_canvas(move, board_size, canvas_size) for move in legal_moves]


def _battlefield_context(board: np.ndarray) -> Tuple[Optional[Tuple[float, float]], int, int, int, int, int]:
    occupied = np.argwhere(board != 0)
    if occupied.size == 0:
        return None, 0, 0, 0, 0

    min_row = int(np.min(occupied[:, 0]))
    max_row = int(np.max(occupied[:, 0]))
    min_col = int(np.min(occupied[:, 1]))
    max_col = int(np.max(occupied[:, 1]))
    center = ((min_row + max_row) / 2.0, (min_col + max_col) / 2.0)
    extent = max(max_row - min_row + 1, max_col - min_col + 1)
    return center, extent, min_row, max_row, min_col, max_col


def _battlefield_move_weights(board: np.ndarray, legal_moves: Sequence[int], board_size: int, last_move: Optional[Tuple[int, int]] = None) -> np.ndarray:
    legal_moves = list(legal_moves)
    if not legal_moves:
        return np.array([], dtype=np.float32)

    if np.count_nonzero(board) == 0:
        return _center_weights(board_size, legal_moves)

    context = _battlefield_context(board)
    if context[0] is None:
        return _center_weights(board_size, legal_moves)

    battlefield_center, extent, min_row, max_row, min_col, max_col = context
    assert battlefield_center is not None
    center_row, center_col = battlefield_center
    local_anchor = last_move if last_move is not None else battlefield_center
    anchor_row, anchor_col = local_anchor
    expanded_margin = max(1, extent // 3)
    outer_row_min = max(0, min_row - expanded_margin)
    outer_row_max = min(board_size - 1, max_row + expanded_margin)
    outer_col_min = max(0, min_col - expanded_margin)
    outer_col_max = min(board_size - 1, max_col + expanded_margin)

    weights: List[float] = []
    for move in legal_moves:
        row, col = divmod(int(move), board_size)
        distance_to_anchor = np.sqrt((row - anchor_row) ** 2 + (col - anchor_col) ** 2)
        distance_to_center = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
        inside_battlefield = outer_row_min <= row <= outer_row_max and outer_col_min <= col <= outer_col_max

        anchor_scale = max(1.0, extent / 2.0)
        center_scale = max(1.0, extent / 1.5)
        score = np.exp(-distance_to_anchor / anchor_scale) * 0.75 + np.exp(-distance_to_center / center_scale) * 0.25
        if inside_battlefield:
            score *= 1.35
        else:
            score *= 0.65
        weights.append(float(score))

    arr = np.asarray(weights, dtype=np.float32)
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        return _center_weights(board_size, legal_moves)
    arr /= total
    return arr


def _heuristic_move_bonus(board: np.ndarray, board_size: int, legal_moves: Sequence[int], last_move: Optional[Tuple[int, int]] = None) -> np.ndarray:
    weights = _battlefield_move_weights(board, legal_moves, board_size, last_move=last_move)
    if weights.size == 0:
        return weights
    return weights


def _count_connected(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    size = board.shape[0]
    count = 0
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
        count += 1
        r += dr
        c += dc
    return count


def _is_winning_move(board: np.ndarray, move: int, board_size: int, player: int, win_length: int) -> bool:
    row, col = divmod(int(move), int(board_size))
    if int(board[row, col]) != 0:
        return False

    temp = board.copy()
    temp[row, col] = int(player)
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dr, dc in directions:
        length = 1
        length += _count_connected(temp, row, col, dr, dc, int(player))
        length += _count_connected(temp, row, col, -dr, -dc, int(player))
        if length >= win_length:
            return True
    return False


def _find_immediate_wins(board: np.ndarray, legal_moves: Sequence[int], board_size: int, player: int, win_length: int) -> List[int]:
    wins: List[int] = []
    for move in legal_moves:
        if _is_winning_move(board, int(move), board_size, player, win_length):
            wins.append(int(move))
    return wins


def _find_open_three_end_blocks(board: np.ndarray, legal_moves: Sequence[int], board_size: int, player: int) -> List[int]:
    legal_set = {int(move) for move in legal_moves}
    if not legal_set:
        return []

    threat_blocks: set[int] = set()
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for row in range(board_size):
        for col in range(board_size):
            if int(board[row, col]) != int(player):
                continue
            for dr, dc in directions:
                prev_r = row - dr
                prev_c = col - dc
                if 0 <= prev_r < board_size and 0 <= prev_c < board_size and int(board[prev_r, prev_c]) == int(player):
                    continue

                r1 = row + dr
                c1 = col + dc
                r2 = row + 2 * dr
                c2 = col + 2 * dc
                if not (0 <= r1 < board_size and 0 <= c1 < board_size and 0 <= r2 < board_size and 0 <= c2 < board_size):
                    continue
                if int(board[r1, c1]) != int(player) or int(board[r2, c2]) != int(player):
                    continue

                left_r = row - dr
                left_c = col - dc
                right_r = row + 3 * dr
                right_c = col + 3 * dc
                if not (0 <= left_r < board_size and 0 <= left_c < board_size and 0 <= right_r < board_size and 0 <= right_c < board_size):
                    continue
                if int(board[left_r, left_c]) != 0 or int(board[right_r, right_c]) != 0:
                    continue

                left_move = left_r * board_size + left_c
                right_move = right_r * board_size + right_c
                if left_move in legal_set:
                    threat_blocks.add(int(left_move))
                if right_move in legal_set:
                    threat_blocks.add(int(right_move))

    return list(threat_blocks)


def _pick_weighted_move(board: np.ndarray, board_size: int, moves: Sequence[int]) -> int:
    moves = list(moves)
    if len(moves) == 1:
        return int(moves[0])
    weights = _heuristic_move_bonus(board, board_size, moves, last_move=None)
    return int(np.random.choice(moves, p=weights))


def _legal_canvas_mask(board_size: int, canvas_size: int = MAX_BOARD_SIZE) -> np.ndarray:
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    offset = _board_offset(board_size, canvas_size)
    mask[offset : offset + board_size, offset : offset + board_size] = 1.0
    return mask.reshape(-1)


def _center_weights(board_size: int, legal_moves: Sequence[int]) -> np.ndarray:
    if not legal_moves:
        return np.array([], dtype=np.float32)
    center = (board_size - 1) / 2.0
    weights = []
    for move in legal_moves:
        row, col = divmod(int(move), board_size)
        distance = np.sqrt((row - center) ** 2 + (col - center) ** 2)
        # Keep all moves possible, but softly favor center and near-center cells.
        weights.append(1.0 / (1.0 + distance))
    arr = np.asarray(weights, dtype=np.float32)
    arr /= np.sum(arr)
    return arr


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
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        activations = x
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weight + bias
            activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
            activations = np.clip(activations, -MAX_ABS_ACTIVATION, MAX_ABS_ACTIVATION)
            if layer_index < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
                activations = np.clip(activations, 0.0, MAX_ABS_ACTIVATION)
        return activations

    def train_batch(self, x: np.ndarray, target: np.ndarray, learning_rate: float) -> float:
        x = np.asarray(x, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.clip(target, -MAX_ABS_Q, MAX_ABS_Q)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target.ndim == 1:
            target = target.reshape(1, -1)

        activations = [x]
        pre_activations = []
        current = x
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
            current = np.clip(current, -MAX_ABS_ACTIVATION, MAX_ABS_ACTIVATION)
            pre_activations.append(current)
            if layer_index < len(self.weights) - 1:
                current = np.maximum(current, 0.0)
                current = np.clip(current, 0.0, MAX_ABS_ACTIVATION)
            activations.append(current)

        prediction = activations[-1]
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        error = prediction - target
        error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        error = np.clip(error, -MAX_ABS_Q, MAX_ABS_Q)
        loss = float(np.mean(error * error))
        if not np.isfinite(loss):
            return 0.0

        gradient = (2.0 / x.shape[0]) * error
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        gradient = np.clip(gradient, -MAX_ABS_GRADIENT, MAX_ABS_GRADIENT)
        weights_snapshot = [weight.copy() for weight in self.weights]
        grad_weights: List[np.ndarray] = [np.zeros_like(weight) for weight in self.weights]
        grad_biases: List[np.ndarray] = [np.zeros_like(bias) for bias in self.biases]

        for layer_index in reversed(range(len(self.weights))):
            activation_prev = activations[layer_index]
            grad_weights[layer_index] = activation_prev.T @ gradient
            grad_biases[layer_index] = gradient.sum(axis=0)
            grad_weights[layer_index] = np.nan_to_num(grad_weights[layer_index], nan=0.0, posinf=0.0, neginf=0.0)
            grad_biases[layer_index] = np.nan_to_num(grad_biases[layer_index], nan=0.0, posinf=0.0, neginf=0.0)
            grad_weights[layer_index] = np.clip(grad_weights[layer_index], -MAX_ABS_GRADIENT, MAX_ABS_GRADIENT)
            grad_biases[layer_index] = np.clip(grad_biases[layer_index], -MAX_ABS_GRADIENT, MAX_ABS_GRADIENT)

            if layer_index > 0:
                gradient = gradient @ weights_snapshot[layer_index].T
                gradient = gradient * (pre_activations[layer_index - 1] > 0.0)
                gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
                gradient = np.clip(gradient, -MAX_ABS_GRADIENT, MAX_ABS_GRADIENT)

        for layer_index in range(len(self.weights)):
            self.weights[layer_index] -= learning_rate * grad_weights[layer_index]
            self.biases[layer_index] -= learning_rate * grad_biases[layer_index]
            self.weights[layer_index] = np.nan_to_num(self.weights[layer_index], nan=0.0, posinf=0.0, neginf=0.0)
            self.biases[layer_index] = np.nan_to_num(self.biases[layer_index], nan=0.0, posinf=0.0, neginf=0.0)
            self.weights[layer_index] = np.clip(self.weights[layer_index], -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
            self.biases[layer_index] = np.clip(self.biases[layer_index], -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)

        return loss


class QLearningAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float = 0.14,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.992,
    ) -> None:
        self.canvas_size = MAX_BOARD_SIZE
        self.board_size = int(board_size)
        self.action_size = self.canvas_size * self.canvas_size
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table: Dict[bytes, np.ndarray] = {}
        self.total_updates = 0

    def set_board_size(self, board_size: int) -> None:
        self.board_size = int(board_size)

    def reset_knowledge(self) -> None:
        self.q_table.clear()
        self.total_updates = 0
        self.epsilon = self.initial_epsilon

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        key = _embed_board(state, self.board_size, self.canvas_size).astype(np.int8).tobytes()
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[key]

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = True) -> int:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return 0

        win_length = min(5, self.board_size)
        winning_moves = _find_immediate_wins(state, legal_moves, self.board_size, player=1, win_length=win_length)
        if winning_moves:
            return _pick_weighted_move(state, self.board_size, winning_moves)

        block_moves = _find_immediate_wins(state, legal_moves, self.board_size, player=-1, win_length=win_length)
        if block_moves:
            return _pick_weighted_move(state, self.board_size, block_moves)

        open_three_blocks = _find_open_three_end_blocks(state, legal_moves, self.board_size, player=-1)
        if open_three_blocks:
            return _pick_weighted_move(state, self.board_size, open_three_blocks)

        legal_canvas_moves = _legal_canvas_actions(legal_moves, self.board_size, self.canvas_size)
        if explore and np.random.random() < self.epsilon:
            weights = _heuristic_move_bonus(state, self.board_size, legal_moves, last_move=None)
            chosen_canvas = int(np.random.choice(legal_canvas_moves, p=weights))
            return int(_canvas_action_to_board(chosen_canvas, self.board_size, self.canvas_size))

        q_values = np.nan_to_num(self._get_q_values(state), nan=0.0, posinf=0.0, neginf=0.0)
        legal_q = q_values[legal_canvas_moves].astype(np.float32)
        if legal_q.size == 0:
            return int(np.random.choice(legal_moves))

        max_q = float(np.max(legal_q))
        candidates = [move for move, value in zip(legal_canvas_moves, legal_q) if np.isfinite(value) and abs(float(value) - max_q) < 1e-6]
        if not candidates:
            heuristic = _heuristic_move_bonus(state, self.board_size, legal_moves, last_move=None)
            blended = legal_q + 0.25 * heuristic
            best_index = int(np.argmax(blended))
            return int(_canvas_action_to_board(legal_canvas_moves[best_index], self.board_size, self.canvas_size))
        if len(candidates) == 1:
            return int(_canvas_action_to_board(candidates[0], self.board_size, self.canvas_size))

        candidate_board_moves = [_canvas_action_to_board(move, self.board_size, self.canvas_size) for move in candidates]
        weights = _heuristic_move_bonus(state, self.board_size, candidate_board_moves, last_move=None)
        chosen_canvas = int(np.random.choice(candidates, p=weights))
        return int(_canvas_action_to_board(chosen_canvas, self.board_size, self.canvas_size))

    def learn_transition(self, transition: Transition) -> float:
        total_loss = 0.0
        board_size = int(transition.board_size or self.board_size)
        state_canvas = _embed_board(transition.state, board_size, self.canvas_size)
        next_state_canvas = _embed_board(transition.next_state, board_size, self.canvas_size)
        for transform_id in range(8):
            state_variant = transform_board(transition.state, transform_id)
            next_state_variant = transform_board(transition.next_state, transform_id)
            action_variant = transform_action(transition.action, board_size, transform_id)
            state_key = transform_board(state_canvas, transform_id).astype(np.int8).tobytes()
            q_values = self.q_table.setdefault(state_key, np.zeros(self.action_size, dtype=np.float32))
            next_q_values = self.q_table.setdefault(transform_board(next_state_canvas, transform_id).astype(np.int8).tobytes(), np.zeros(self.action_size, dtype=np.float32))
            action_canvas = _board_action_to_canvas(action_variant, board_size, self.canvas_size)

            if transition.done:
                target_value = transition.reward
            else:
                board_region = _legal_canvas_mask(board_size, self.canvas_size)
                legal_next = np.flatnonzero((transform_board(next_state_canvas, transform_id).reshape(-1) == 0) & (board_region > 0))
                if legal_next.size == 0:
                    target_value = transition.reward
                else:
                    target_value = transition.reward + self.gamma * float(np.max(next_q_values[legal_next]))

            prediction = float(q_values[action_canvas])
            updated_q = prediction + self.learning_rate * (target_value - prediction)
            q_values[action_canvas] = _safe_clip_scalar(updated_q, MAX_ABS_Q)
            total_loss += abs(target_value - prediction)

        self.total_updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss / 8.0

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "q_learning",
            "board_size": self.board_size,
            "canvas_size": self.canvas_size,
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
        agent.canvas_size = payload.get("canvas_size", MAX_BOARD_SIZE)
        agent.action_size = agent.canvas_size * agent.canvas_size
        agent.total_updates = payload.get("total_updates", 0)
        agent.q_table = payload["q_table"]
        return agent


class DQNAgent:
    def __init__(
        self,
        board_size: int,
        learning_rate: float = 0.08,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.996,
        memory_size: int = 50_000,
        batch_size: int = 48,
        hidden_layers: Sequence[int] = (256, 128),
        target_update_every: int = 120,
        replay_loops: int = 2,
    ) -> None:
        self.canvas_size = MAX_BOARD_SIZE
        self.board_size = int(board_size)
        self.action_size = self.canvas_size * self.canvas_size
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.replay_loops = int(max(1, replay_loops))
        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.memory: Deque[Transition] = deque(maxlen=int(memory_size))
        self.model = DenseNetwork(self.action_size, self.hidden_layers, self.action_size)
        self.target_model = self.model.clone()
        self.train_steps = 0
        self.total_updates = 0

    def set_board_size(self, board_size: int) -> None:
        self.board_size = int(board_size)

    def reset_knowledge(self) -> None:
        self.memory.clear()
        self.model = DenseNetwork(self.action_size, self.hidden_layers, self.action_size)
        self.target_model = self.model.clone()
        self.train_steps = 0
        self.total_updates = 0
        self.epsilon = self.initial_epsilon

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = True) -> int:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return 0

        win_length = min(5, self.board_size)
        winning_moves = _find_immediate_wins(state, legal_moves, self.board_size, player=1, win_length=win_length)
        if winning_moves:
            return _pick_weighted_move(state, self.board_size, winning_moves)

        block_moves = _find_immediate_wins(state, legal_moves, self.board_size, player=-1, win_length=win_length)
        if block_moves:
            return _pick_weighted_move(state, self.board_size, block_moves)

        open_three_blocks = _find_open_three_end_blocks(state, legal_moves, self.board_size, player=-1)
        if open_three_blocks:
            return _pick_weighted_move(state, self.board_size, open_three_blocks)

        legal_canvas_moves = _legal_canvas_actions(legal_moves, self.board_size, self.canvas_size)
        if explore and np.random.random() < self.epsilon:
            weights = _heuristic_move_bonus(state, self.board_size, legal_moves, last_move=None)
            chosen_canvas = int(np.random.choice(legal_canvas_moves, p=weights))
            return int(_canvas_action_to_board(chosen_canvas, self.board_size, self.canvas_size))

        q_values = np.nan_to_num(self.model.predict(_state_to_vector(state, self.board_size))[0], nan=0.0, posinf=0.0, neginf=0.0)
        q_values = np.clip(q_values, -MAX_ABS_Q, MAX_ABS_Q)
        legal_q = q_values[legal_canvas_moves].astype(np.float32)
        if legal_q.size == 0:
            return int(np.random.choice(legal_moves))

        max_q = float(np.max(legal_q))
        candidates = [move for move, value in zip(legal_canvas_moves, legal_q) if np.isfinite(value) and abs(float(value) - max_q) < 1e-6]
        if not candidates:
            heuristic = _heuristic_move_bonus(state, self.board_size, legal_moves, last_move=None)
            blended = legal_q + 0.25 * heuristic
            best_index = int(np.argmax(blended))
            return int(_canvas_action_to_board(legal_canvas_moves[best_index], self.board_size, self.canvas_size))
        if len(candidates) == 1:
            return int(_canvas_action_to_board(candidates[0], self.board_size, self.canvas_size))

        candidate_board_moves = [_canvas_action_to_board(move, self.board_size, self.canvas_size) for move in candidates]
        weights = _heuristic_move_bonus(state, self.board_size, candidate_board_moves, last_move=None)
        chosen_canvas = int(np.random.choice(candidates, p=weights))
        return int(_canvas_action_to_board(chosen_canvas, self.board_size, self.canvas_size))

    def remember(self, transition: Transition) -> None:
        board_size = int(transition.board_size or self.board_size)
        state_canvas = _embed_board(transition.state, board_size, self.canvas_size)
        next_state_canvas = _embed_board(transition.next_state, board_size, self.canvas_size)
        for transform_id in range(8):
            state_variant = transform_board(state_canvas, transform_id)
            next_state_variant = transform_board(next_state_canvas, transform_id)
            action_variant = _board_action_to_canvas(transform_action(transition.action, board_size, transform_id), board_size, self.canvas_size)
            self.memory.append(
                Transition(
                    state=state_variant.astype(np.float32),
                    action=int(action_variant),
                    reward=float(transition.reward),
                    next_state=next_state_variant.astype(np.float32),
                    done=bool(transition.done),
                    board_size=board_size,
                )
            )

    def learn_from_transition(self, transition: Transition) -> float:
        self.remember(transition)
        total = 0.0
        for _ in range(self.replay_loops):
            total += self.replay()
        return total / float(self.replay_loops)

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return 0.0

        batch = [self.memory[index] for index in np.random.choice(len(self.memory), self.batch_size, replace=False)]
        states = np.vstack([item.state.reshape(1, -1) for item in batch])
        next_states = np.vstack([item.next_state.reshape(1, -1) for item in batch])
        predicted = self.model.predict(states)
        predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
        predicted = np.clip(predicted, -MAX_ABS_Q, MAX_ABS_Q)
        targets = predicted.copy()
        next_q_values = self.target_model.predict(next_states)
        next_q_values = np.nan_to_num(next_q_values, nan=0.0, posinf=0.0, neginf=0.0)
        next_q_values = np.clip(next_q_values, -MAX_ABS_Q, MAX_ABS_Q)

        for row_index, item in enumerate(batch):
            board_region = _legal_canvas_mask(int(item.board_size or self.board_size), self.canvas_size)
            if item.done:
                target_value = item.reward
            else:
                legal_next = np.flatnonzero((item.next_state.reshape(-1) == 0) & (board_region > 0))
                if legal_next.size == 0:
                    target_value = item.reward
                else:
                    target_value = item.reward + self.gamma * float(np.max(next_q_values[row_index, legal_next]))
            targets[row_index, item.action] = _safe_clip_scalar(target_value, MAX_ABS_Q)

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
            "canvas_size": self.canvas_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "hidden_layers": self.hidden_layers,
            "target_update_every": self.target_update_every,
            "replay_loops": self.replay_loops,
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
            replay_loops=payload.get("replay_loops", 2),
        )
        agent.canvas_size = payload.get("canvas_size", MAX_BOARD_SIZE)
        agent.action_size = agent.canvas_size * agent.canvas_size
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
    learning_rate: Optional[float] = None,
    gamma: float = 0.9,
) -> object:
    algorithm = algorithm.lower().strip()
    if algorithm == "auto":
        algorithm = "q_learning" if board_size <= 5 else "dqn"
    if algorithm in {"q", "q_learning", "q-learning"}:
        lr = 0.14 if learning_rate is None else float(learning_rate)
        return QLearningAgent(board_size=board_size, learning_rate=lr, gamma=gamma)
    if algorithm == "dqn":
        lr = 0.08 if learning_rate is None else float(learning_rate)
        return DQNAgent(board_size=board_size, learning_rate=lr, gamma=gamma)
    raise ValueError(f"Unknown algorithm: {algorithm}")
