from __future__ import annotations

import pickle
import importlib
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


def _try_import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception:  # pragma: no cover - optional dependency guard
        return None


_TORCH = _try_import_torch()
TORCH_AVAILABLE = _TORCH is not None

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
        return None, 0, 0, 0, 0, 0

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


@lru_cache(maxsize=16)
def _line_index_cache(board_size: int) -> Tuple[Tuple[int, ...], ...]:
    lines: List[Tuple[int, ...]] = []

    for row in range(board_size):
        lines.append(tuple(row * board_size + col for col in range(board_size)))
    for col in range(board_size):
        lines.append(tuple(row * board_size + col for row in range(board_size)))

    for start_col in range(board_size):
        diagonal: List[int] = []
        row = 0
        col = start_col
        while row < board_size and col < board_size:
            diagonal.append(row * board_size + col)
            row += 1
            col += 1
        if len(diagonal) >= 5:
            lines.append(tuple(diagonal))
    for start_row in range(1, board_size):
        diagonal = []
        row = start_row
        col = 0
        while row < board_size and col < board_size:
            diagonal.append(row * board_size + col)
            row += 1
            col += 1
        if len(diagonal) >= 5:
            lines.append(tuple(diagonal))

    for start_col in range(board_size):
        diagonal = []
        row = 0
        col = start_col
        while row < board_size and col >= 0:
            diagonal.append(row * board_size + col)
            row += 1
            col -= 1
        if len(diagonal) >= 5:
            lines.append(tuple(diagonal))
    for start_row in range(1, board_size):
        diagonal = []
        row = start_row
        col = board_size - 1
        while row < board_size and col >= 0:
            diagonal.append(row * board_size + col)
            row += 1
            col -= 1
        if len(diagonal) >= 5:
            lines.append(tuple(diagonal))

    return tuple(lines)


def _find_one_move_wins(board: np.ndarray, legal_moves: Sequence[int], board_size: int, player: int, win_length: int) -> List[int]:
    legal_set: Set[int] = {int(move) for move in legal_moves}
    if not legal_set:
        return []

    flat = board.reshape(-1)
    winners: Set[int] = set()
    for line in _line_index_cache(board_size):
        if len(line) < win_length:
            continue
        values = [int(flat[index]) for index in line]
        limit = len(line) - win_length + 1
        for start in range(limit):
            segment = values[start : start + win_length]
            player_count = 0
            empty_count = 0
            empty_pos = -1
            blocked = False
            for offset, cell in enumerate(segment):
                if cell == int(player):
                    player_count += 1
                elif cell == 0:
                    empty_count += 1
                    empty_pos = offset
                else:
                    blocked = True
                    break
            if blocked or player_count != win_length - 1 or empty_count != 1:
                continue
            move = int(line[start + empty_pos])
            if move in legal_set:
                winners.add(move)

    return list(winners)


def _frontier_moves(board: np.ndarray, legal_moves: Sequence[int], board_size: int, radius: int = 2) -> List[int]:
    legal = [int(move) for move in legal_moves]
    if not legal:
        return []
    if int(np.count_nonzero(board)) == 0:
        return legal

    legal_set = set(legal)
    occupied = np.argwhere(board != 0)
    frontier: Set[int] = set()
    for row, col in occupied:
        r0 = max(0, int(row) - radius)
        r1 = min(board_size - 1, int(row) + radius)
        c0 = max(0, int(col) - radius)
        c1 = min(board_size - 1, int(col) + radius)
        for r in range(r0, r1 + 1):
            base = r * board_size
            for c in range(c0, c1 + 1):
                move = base + c
                if move in legal_set:
                    frontier.add(move)

    if len(frontier) >= 4:
        return list(frontier)
    return legal


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
    return _find_one_move_wins(board, legal_moves, board_size, player, win_length)


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


def _line_length_open_ends(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> Tuple[int, int]:
    size = board.shape[0]

    forward = 0
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
        forward += 1
        r += dr
        c += dc
    open_forward = 1 if (0 <= r < size and 0 <= c < size and int(board[r, c]) == 0) else 0

    backward = 0
    r = row - dr
    c = col - dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
        backward += 1
        r -= dr
        c -= dc
    open_backward = 1 if (0 <= r < size and 0 <= c < size and int(board[r, c]) == 0) else 0

    return 1 + forward + backward, open_forward + open_backward


def _max_line_after_move(board: np.ndarray, move: int, board_size: int, player: int) -> Tuple[int, int]:
    row, col = divmod(int(move), board_size)
    best_len = 1
    best_open = 0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        length, open_ends = _line_length_open_ends(board, row, col, dr, dc, player)
        if length > best_len or (length == best_len and open_ends > best_open):
            best_len = length
            best_open = open_ends
    return best_len, best_open


def _choose_tactical_move(board: np.ndarray, legal_moves: Sequence[int], board_size: int) -> Optional[int]:
    legal_moves = [int(move) for move in legal_moves]
    if not legal_moves:
        return None

    win_length = min(5, board_size)

    winning_moves = _find_immediate_wins(board, legal_moves, board_size, player=1, win_length=win_length)
    if winning_moves:
        return _pick_weighted_move(board, board_size, winning_moves)

    block_moves = _find_immediate_wins(board, legal_moves, board_size, player=-1, win_length=win_length)
    if block_moves:
        return _pick_weighted_move(board, board_size, block_moves)

    open_three_blocks = _find_open_three_end_blocks(board, legal_moves, board_size, player=-1)
    if open_three_blocks:
        return _pick_weighted_move(board, board_size, open_three_blocks)

    candidate_moves = _frontier_moves(board, legal_moves, board_size, radius=2)
    heuristic = _heuristic_move_bonus(board, board_size, candidate_moves, last_move=None)
    h_map = {int(move): float(weight) for move, weight in zip(candidate_moves, heuristic)}
    move_scores: Dict[int, float] = {}

    for move in candidate_moves:
        temp = board.copy()
        row, col = divmod(move, board_size)
        if int(temp[row, col]) != 0:
            continue
        temp[row, col] = 1
        next_legal = [candidate for candidate in candidate_moves if candidate != move and int(temp[candidate // board_size, candidate % board_size]) == 0]
        if len(next_legal) < 4:
            next_legal = [candidate for candidate in legal_moves if candidate != move]

        own_next_wins = _find_immediate_wins(temp, next_legal, board_size, player=1, win_length=win_length)
        opp_next_wins = _find_immediate_wins(temp, next_legal, board_size, player=-1, win_length=win_length)

        score = 0.0
        if len(own_next_wins) >= 2:
            score += 9000.0
        elif len(own_next_wins) == 1:
            score += 2500.0

        if len(opp_next_wins) == 0:
            score += 320.0
        else:
            score -= 700.0 * float(len(opp_next_wins))

        own_open_three = len(_find_open_three_end_blocks(temp, next_legal, board_size, player=1))
        opp_open_three = len(_find_open_three_end_blocks(temp, next_legal, board_size, player=-1))
        score += 50.0 * float(own_open_three)
        score -= 70.0 * float(opp_open_three)

        max_len, open_ends = _max_line_after_move(temp, move, board_size, player=1)
        score += 20.0 * float(max_len)
        score += 10.0 * float(open_ends)
        if max_len >= 4:
            score += 180.0
        elif max_len == 3 and open_ends == 2:
            score += 120.0

        score += 35.0 * h_map.get(move, 0.0)
        move_scores[move] = score

    if not move_scores:
        return None

    best_score = max(move_scores.values())
    # Keep RL in control when no tactical signal is meaningful.
    if best_score < 140.0:
        return None

    best_moves = [move for move, value in move_scores.items() if abs(value - best_score) < 1e-6]
    if len(best_moves) == 1:
        return int(best_moves[0])

    return _pick_weighted_move(board, board_size, best_moves)


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


def _build_torch_mlp(input_dim: int, hidden_layers: Sequence[int], output_dim: int, torch_mod):
    nn_mod = torch_mod.nn

    class _TorchMLP(nn_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            sizes = [int(input_dim), *[int(width) for width in hidden_layers], int(output_dim)]
            self.linears = nn_mod.ModuleList([nn_mod.Linear(left, right) for left, right in zip(sizes[:-1], sizes[1:])])

        def forward(self, x):
            out = x
            for index, layer in enumerate(self.linears):
                out = layer(out)
                if index < len(self.linears) - 1:
                    out = torch_mod.relu(out)
            return out

    return _TorchMLP()


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
        loss_boost: float = 1.45,
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
        self.loss_boost = float(max(1.0, loss_boost))
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

        tactical = _choose_tactical_move(state, legal_moves, self.board_size)
        if tactical is not None:
            return int(tactical)

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
            effective_lr = self.learning_rate
            if float(transition.reward) < 0.0:
                effective_lr = min(1.0, self.learning_rate * self.loss_boost)
            updated_q = prediction + effective_lr * (target_value - prediction)
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
            "loss_boost": self.loss_boost,
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
            loss_boost=payload.get("loss_boost", 1.45),
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
        hard_negative_ratio: float = 0.45,
        hard_negative_reward_cutoff: float = -8.0,
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
        self.hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 0.9))
        self.hard_negative_reward_cutoff = float(hard_negative_reward_cutoff)
        self.hidden_layers = tuple(int(layer) for layer in hidden_layers)
        self.memory: Deque[Transition] = deque(maxlen=int(memory_size))
        self.hard_negative_memory: Deque[Transition] = deque(maxlen=max(256, int(memory_size // 3)))
        self._torch = _TORCH
        self.backend = "torch" if TORCH_AVAILABLE else "numpy"
        self.device = "cpu"
        if self.backend == "torch":
            torch_mod = self._torch
            assert torch_mod is not None
            self.device = "cuda" if torch_mod.cuda.is_available() else "cpu"
            self.model_torch = _build_torch_mlp(self.action_size, self.hidden_layers, self.action_size, torch_mod).to(self.device)
            self.target_model_torch = _build_torch_mlp(self.action_size, self.hidden_layers, self.action_size, torch_mod).to(self.device)
            self.target_model_torch.load_state_dict(self.model_torch.state_dict())
            self.optimizer = torch_mod.optim.Adam(self.model_torch.parameters(), lr=self.learning_rate)
            self.model = None
            self.target_model = None
        else:
            self.model = DenseNetwork(self.action_size, self.hidden_layers, self.action_size)
            self.target_model = self.model.clone()
            self.model_torch = None
            self.target_model_torch = None
            self.optimizer = None
        self.train_steps = 0
        self.total_updates = 0

    def set_board_size(self, board_size: int) -> None:
        self.board_size = int(board_size)

    def reset_knowledge(self) -> None:
        self.memory.clear()
        self.hard_negative_memory.clear()
        if self.backend == "torch":
            torch_mod = self._torch
            assert torch_mod is not None
            self.model_torch = _build_torch_mlp(self.action_size, self.hidden_layers, self.action_size, torch_mod).to(self.device)
            self.target_model_torch = _build_torch_mlp(self.action_size, self.hidden_layers, self.action_size, torch_mod).to(self.device)
            self.target_model_torch.load_state_dict(self.model_torch.state_dict())
            self.optimizer = torch_mod.optim.Adam(self.model_torch.parameters(), lr=self.learning_rate)
        else:
            self.model = DenseNetwork(self.action_size, self.hidden_layers, self.action_size)
            self.target_model = self.model.clone()
        self.train_steps = 0
        self.total_updates = 0
        self.epsilon = self.initial_epsilon

    def backend_info(self) -> str:
        return f"{self.backend}:{self.device}" if self.backend == "torch" else "numpy:cpu"

    def _predict_batch(self, states: np.ndarray, use_target: bool = False) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        if states.ndim == 1:
            states = states.reshape(1, -1)

        if self.backend == "torch":
            torch_mod = self._torch
            assert torch_mod is not None
            model = self.target_model_torch if use_target else self.model_torch
            assert model is not None
            with torch_mod.no_grad():
                tensor = torch_mod.from_numpy(states).to(self.device)
                outputs = model(tensor)
                outputs = torch_mod.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
                outputs = torch_mod.clamp(outputs, -MAX_ABS_Q, MAX_ABS_Q)
            return outputs.detach().cpu().numpy().astype(np.float32)

        assert self.model is not None
        return self.model.predict(states)

    def _train_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        states = np.asarray(states, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.clip(targets, -MAX_ABS_Q, MAX_ABS_Q)

        if self.backend == "torch":
            torch_mod = self._torch
            assert torch_mod is not None
            assert self.model_torch is not None
            assert self.optimizer is not None
            input_tensor = torch_mod.from_numpy(states).to(self.device)
            target_tensor = torch_mod.from_numpy(targets).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model_torch(input_tensor)
            prediction = torch_mod.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
            prediction = torch_mod.clamp(prediction, -MAX_ABS_Q, MAX_ABS_Q)
            loss = torch_mod.nn.functional.mse_loss(prediction, target_tensor)
            if not torch_mod.isfinite(loss):
                return 0.0
            loss.backward()
            torch_mod.nn.utils.clip_grad_norm_(self.model_torch.parameters(), max_norm=MAX_ABS_GRADIENT)
            self.optimizer.step()
            return float(loss.detach().cpu().item())

        assert self.model is not None
        return self.model.train_batch(states, targets, self.learning_rate)

    def _sync_target(self) -> None:
        if self.backend == "torch":
            assert self.model_torch is not None
            assert self.target_model_torch is not None
            self.target_model_torch.load_state_dict(self.model_torch.state_dict())
        else:
            assert self.model is not None
            self.target_model = self.model.clone()

    def _load_numpy_weights_into_torch(
        self,
        model_weights: Sequence[np.ndarray],
        model_biases: Sequence[np.ndarray],
        target_weights: Sequence[np.ndarray],
        target_biases: Sequence[np.ndarray],
    ) -> None:
        if self.backend != "torch":
            return
        torch_mod = self._torch
        assert torch_mod is not None
        assert self.model_torch is not None
        assert self.target_model_torch is not None

        for layer, weight, bias in zip(self.model_torch.linears, model_weights, model_biases):
            layer.weight.data.copy_(torch_mod.from_numpy(np.asarray(weight, dtype=np.float32).T).to(self.device))
            layer.bias.data.copy_(torch_mod.from_numpy(np.asarray(bias, dtype=np.float32)).to(self.device))
        for layer, weight, bias in zip(self.target_model_torch.linears, target_weights, target_biases):
            layer.weight.data.copy_(torch_mod.from_numpy(np.asarray(weight, dtype=np.float32).T).to(self.device))
            layer.bias.data.copy_(torch_mod.from_numpy(np.asarray(bias, dtype=np.float32)).to(self.device))

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = True) -> int:
        legal_moves = list(legal_moves)
        if not legal_moves:
            return 0

        tactical = _choose_tactical_move(state, legal_moves, self.board_size)
        if tactical is not None:
            return int(tactical)

        legal_canvas_moves = _legal_canvas_actions(legal_moves, self.board_size, self.canvas_size)
        if explore and np.random.random() < self.epsilon:
            weights = _heuristic_move_bonus(state, self.board_size, legal_moves, last_move=None)
            chosen_canvas = int(np.random.choice(legal_canvas_moves, p=weights))
            return int(_canvas_action_to_board(chosen_canvas, self.board_size, self.canvas_size))

        q_values = np.nan_to_num(self._predict_batch(_state_to_vector(state, self.board_size))[0], nan=0.0, posinf=0.0, neginf=0.0)
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
            if float(transition.reward) <= self.hard_negative_reward_cutoff:
                self.hard_negative_memory.append(
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

        hard_count = int(round(self.batch_size * self.hard_negative_ratio))
        hard_count = min(hard_count, len(self.hard_negative_memory), self.batch_size)
        regular_count = self.batch_size - hard_count

        batch: List[Transition] = []
        if hard_count > 0:
            hard_indices = np.random.choice(len(self.hard_negative_memory), hard_count, replace=False)
            batch.extend(self.hard_negative_memory[index] for index in hard_indices)
        if regular_count > 0:
            regular_indices = np.random.choice(len(self.memory), regular_count, replace=False)
            batch.extend(self.memory[index] for index in regular_indices)

        if len(batch) < self.batch_size:
            refill = self.batch_size - len(batch)
            refill_indices = np.random.choice(len(self.memory), refill, replace=False)
            batch.extend(self.memory[index] for index in refill_indices)
        states = np.vstack([item.state.reshape(1, -1) for item in batch])
        next_states = np.vstack([item.next_state.reshape(1, -1) for item in batch])
        predicted = self._predict_batch(states, use_target=False)
        predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
        predicted = np.clip(predicted, -MAX_ABS_Q, MAX_ABS_Q)
        targets = predicted.copy()
        next_q_values = self._predict_batch(next_states, use_target=True)
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

        loss = self._train_batch(states, targets)
        self.train_steps += 1
        self.total_updates += 1
        if self.train_steps % self.target_update_every == 0:
            self._sync_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def save(self, path: str | Path) -> None:
        model_weights = None
        model_biases = None
        target_weights = None
        target_biases = None
        if self.backend == "torch":
            assert self.model_torch is not None
            assert self.target_model_torch is not None
            model_state = {key: value.detach().cpu() for key, value in self.model_torch.state_dict().items()}
            target_state = {key: value.detach().cpu() for key, value in self.target_model_torch.state_dict().items()}
        else:
            assert self.model is not None
            assert self.target_model is not None
            model_state = None
            target_state = None
            model_weights = [weight.copy() for weight in self.model.weights]
            model_biases = [bias.copy() for bias in self.model.biases]
            target_weights = [weight.copy() for weight in self.target_model.weights]
            target_biases = [bias.copy() for bias in self.target_model.biases]

        payload = {
            "type": "dqn",
            "backend": self.backend,
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
            "hard_negative_ratio": self.hard_negative_ratio,
            "hard_negative_reward_cutoff": self.hard_negative_reward_cutoff,
            "train_steps": self.train_steps,
            "total_updates": self.total_updates,
            "device": self.device,
            "torch_model_state": model_state,
            "torch_target_state": target_state,
            "model_weights": model_weights,
            "model_biases": model_biases,
            "target_weights": target_weights,
            "target_biases": target_biases,
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
            hard_negative_ratio=payload.get("hard_negative_ratio", 0.45),
            hard_negative_reward_cutoff=payload.get("hard_negative_reward_cutoff", -8.0),
        )
        agent.canvas_size = payload.get("canvas_size", MAX_BOARD_SIZE)
        agent.action_size = agent.canvas_size * agent.canvas_size
        agent.train_steps = payload.get("train_steps", 0)
        agent.total_updates = payload.get("total_updates", 0)

        payload_backend = payload.get("backend", "numpy")
        if payload_backend == "torch" and agent.backend == "torch":
            assert agent.model_torch is not None
            assert agent.target_model_torch is not None
            state = payload.get("torch_model_state")
            target_state = payload.get("torch_target_state")
            if state is not None and target_state is not None:
                agent.model_torch.load_state_dict(state)
                agent.target_model_torch.load_state_dict(target_state)
        elif payload.get("model_weights") is not None:
            if agent.backend == "torch":
                agent._load_numpy_weights_into_torch(
                    payload["model_weights"],
                    payload["model_biases"],
                    payload["target_weights"],
                    payload["target_biases"],
                )
            else:
                assert agent.model is not None
                assert agent.target_model is not None
                agent.model.weights = [weight.copy() for weight in payload["model_weights"]]
                agent.model.biases = [bias.copy() for bias in payload["model_biases"]]
                agent.target_model.weights = [weight.copy() for weight in payload["target_weights"]]
                agent.target_model.biases = [bias.copy() for bias in payload["target_biases"]]
        else:
            raise ValueError("Unsupported DQN checkpoint format")

        if agent.backend == "torch" and agent.optimizer is not None and agent.model_torch is not None:
            torch_mod = agent._torch
            assert torch_mod is not None
            agent.optimizer = torch_mod.optim.Adam(agent.model_torch.parameters(), lr=agent.learning_rate)
        return agent


def create_agent(
    board_size: int,
    algorithm: str = "auto",
    learning_rate: Optional[float] = None,
    gamma: float = 0.9,
) -> "QLearningAgent | DQNAgent":
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
