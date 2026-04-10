from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StepResult:
    board: np.ndarray
    reward: float
    done: bool
    info: Dict[str, object]


class GomokuEnv:
    """Fast Gomoku environment with reward shaping for RL training.

    The environment keeps the board state as a NumPy array with values:
    - 1 for player one
    - -1 for player two
    - 0 for an empty cell

    Win length is adaptive for curriculum learning:
    - 3x3 -> 3 in a row
    - 4x4 -> 4 in a row
    - 5x5 and above -> 5 in a row
    """

    def __init__(self, board_size: int = 15, win_length: Optional[int] = None) -> None:
        self._window_cache: Dict[Tuple[int, int], List[List[Tuple[int, int]]]] = {}
        self.set_board_size(board_size, win_length)

    def set_board_size(self, board_size: int, win_length: Optional[int] = None) -> None:
        self.board_size = int(board_size)
        self.win_length = int(win_length) if win_length is not None else min(5, self.board_size)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move: Optional[Tuple[int, int]] = None
        self.last_win_line: Optional[List[Tuple[int, int]]] = None
        self.done = False
        self.winner = 0
        self.move_count = 0
        self._windows = self._get_windows(self.board_size, self.win_length)

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None
        self.last_win_line = None
        self.done = False
        self.winner = 0
        self.move_count = 0
        return self.board.copy()

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def get_perspective_board(self, player: int) -> np.ndarray:
        return (self.board * int(player)).copy()

    def legal_moves(self) -> List[int]:
        empty = np.flatnonzero(self.board.reshape(-1) == 0)
        return empty.tolist()

    def legal_moves_mask(self) -> np.ndarray:
        mask = (self.board.reshape(-1) == 0).astype(np.float32)
        return mask

    def is_valid_move(self, action: int) -> bool:
        row, col = divmod(int(action), self.board_size)
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.get_state(), 0.0, True, {"terminated": True, "winner": self.winner})

        action = int(action)
        player = self.current_player
        row, col = divmod(action, self.board_size)

        if not self.is_valid_move(action):
            self.current_player *= -1
            return StepResult(
                self.get_state(),
                -50.0,
                False,
                {"illegal": True, "player": player, "move": (row, col)},
            )

        before_board = self.board.copy()
        self.board[row, col] = player
        self.last_move = (row, col)
        self.move_count += 1

        reward = 0.0
        done = False
        winner = 0

        win_line = self._find_winning_line(row, col, player)
        if win_line is not None:
            reward = 100.0
            done = True
            winner = player
            self.done = True
            self.winner = player
            self.last_win_line = win_line
        else:
            reward = self._shape_reward(before_board, self.board, player)
            if self.move_count >= self.board_size * self.board_size:
                done = True
                self.done = True
                self.winner = 0
            self.last_win_line = None

        if not done:
            self.current_player *= -1

        info = {
            "illegal": False,
            "player": player,
            "move": (row, col),
            "winner": winner,
            "win_line": self.last_win_line,
            "shaped_reward": reward,
            "done_reason": "win" if winner else ("draw" if done else "running"),
        }
        return StepResult(self.get_state(), float(reward), done, info)

    def _shape_reward(self, board_before: np.ndarray, board_after: np.ndarray, player: int) -> float:
        own_before = self._count_threats(board_before, player)
        own_after = self._count_threats(board_after, player)
        opp_before = self._count_threats(board_before, -player)
        opp_after = self._count_threats(board_after, -player)

        reward = 0.0
        reward += 25.0 * max(0, own_after[4] - own_before[4])
        reward += 15.0 * max(0, own_after[3] - own_before[3])
        reward += 2.0 * max(0, own_after[2] - own_before[2])
        reward += 40.0 * max(0, opp_before[4] - opp_after[4])
        reward += 20.0 * max(0, opp_before[3] - opp_after[3])
        return float(reward)

    def _count_threats(self, board: np.ndarray, player: int) -> Dict[int, int]:
        counts = {2: 0, 3: 0, 4: 0}
        opponent = -player

        for window in self._windows:
            values = [int(board[r, c]) for r, c in window]
            if opponent in values:
                continue
            stones = values.count(player)
            if stones in counts:
                counts[stones] += 1
        return counts

    def _find_winning_line(self, row: int, col: int, player: int) -> Optional[List[Tuple[int, int]]]:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            negative = self._collect_one_direction(row, col, -dr, -dc, player)
            positive = self._collect_one_direction(row, col, dr, dc, player)
            line = list(reversed(negative)) + [(row, col)] + positive
            if len(line) >= self.win_length:
                return line[: self.win_length]
        return None

    def _count_one_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        r = row + dr
        c = col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def _collect_one_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        r = row + dr
        c = col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            cells.append((r, c))
            r += dr
            c += dc
        return cells

    def _get_windows(self, board_size: int, win_length: int) -> List[List[Tuple[int, int]]]:
        cache_key = (board_size, win_length)
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        windows: List[List[Tuple[int, int]]] = []
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            for row in range(board_size):
                for col in range(board_size):
                    end_row = row + (win_length - 1) * dr
                    end_col = col + (win_length - 1) * dc
                    if not (0 <= end_row < board_size and 0 <= end_col < board_size):
                        continue
                    window = [(row + i * dr, col + i * dc) for i in range(win_length)]
                    windows.append(window)

        self._window_cache[cache_key] = windows
        return windows

    def render_text(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[int(cell)] for cell in row) for row in self.board]
        return "\n".join(rows)
