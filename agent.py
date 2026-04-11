from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _count_connected(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    size = board.shape[0]
    count = 0
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == int(player):
        count += 1
        r += dr
        c += dc
    return count


def _frontier_moves(board: np.ndarray, legal_moves: Sequence[int], board_size: int, radius: int = 2) -> List[int]:
    legal = [int(move) for move in legal_moves]
    if not legal:
        return []
    if int(np.count_nonzero(board)) == 0:
        return legal

    legal_set = set(legal)
    occupied = np.argwhere(board != 0)
    frontier = set()
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


def _line_length_open_ends(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> Tuple[int, int]:
    size = board.shape[0]

    forward = 0
    r = row + dr
    c = col + dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == int(player):
        forward += 1
        r += dr
        c += dc
    open_forward = 1 if (0 <= r < size and 0 <= c < size and int(board[r, c]) == 0) else 0

    backward = 0
    r = row - dr
    c = col - dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == int(player):
        backward += 1
        r -= dr
        c -= dc
    open_backward = 1 if (0 <= r < size and 0 <= c < size and int(board[r, c]) == 0) else 0

    return 1 + forward + backward, open_forward + open_backward


class MinimaxAgent:
    def __init__(
        self,
        board_size: int,
        max_depth: int = 3,
        max_branching: int = 14,
    ) -> None:
        self.board_size = int(board_size)
        self.max_depth = max(1, int(max_depth))
        self.max_branching = max(4, int(max_branching))
        self.epsilon = 0.0
        self.total_updates = 0

    def set_board_size(self, board_size: int) -> None:
        self.board_size = int(board_size)

    def reset_knowledge(self) -> None:
        self.total_updates = 0

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "minimax",
            "board_size": self.board_size,
            "max_depth": self.max_depth,
            "max_branching": self.max_branching,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "MinimaxAgent":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(
            board_size=int(payload.get("board_size", 15)),
            max_depth=int(payload.get("max_depth", 3)),
            max_branching=int(payload.get("max_branching", 14)),
        )

    def choose_action(self, state: np.ndarray, legal_moves: Sequence[int], explore: bool = False) -> int:
        del explore
        legal = [int(move) for move in legal_moves]
        if not legal:
            return 0

        candidates = self._rank_candidates(state, legal, player=1)
        if not candidates:
            return int(np.random.choice(legal))

        alpha = -float("inf")
        beta = float("inf")
        best_score = -float("inf")
        best_moves: List[int] = []
        depth = max(1, int(self.max_depth))

        for move in candidates:
            next_board = state.copy()
            row, col = divmod(int(move), self.board_size)
            next_board[row, col] = 1
            score = self._alphabeta(next_board, depth - 1, alpha, beta, maximizing=False)
            if score > best_score + 1e-6:
                best_score = score
                best_moves = [int(move)]
            elif abs(score - best_score) < 1e-6:
                best_moves.append(int(move))
            alpha = max(alpha, best_score)

        if len(best_moves) == 1:
            return int(best_moves[0])
        return int(np.random.choice(best_moves))

    def _alphabeta(self, board: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        terminal_score = self._terminal_score(board, depth)
        if terminal_score is not None:
            return terminal_score

        if depth <= 0:
            return self._evaluate_board(board)

        legal = self._legal_moves(board)
        if not legal:
            return 0.0

        player = 1 if maximizing else -1
        candidates = self._rank_candidates(board, legal, player)
        if maximizing:
            value = -float("inf")
            for move in candidates:
                row, col = divmod(int(move), self.board_size)
                board[row, col] = 1
                value = max(value, self._alphabeta(board, depth - 1, alpha, beta, maximizing=False))
                board[row, col] = 0
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        value = float("inf")
        for move in candidates:
            row, col = divmod(int(move), self.board_size)
            board[row, col] = -1
            value = min(value, self._alphabeta(board, depth - 1, alpha, beta, maximizing=True))
            board[row, col] = 0
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def _terminal_score(self, board: np.ndarray, depth: int) -> Optional[float]:
        win_length = min(5, self.board_size)
        if self._has_win(board, player=1, win_length=win_length):
            return 1_000_000.0 + float(depth)
        if self._has_win(board, player=-1, win_length=win_length):
            return -1_000_000.0 - float(depth)
        if int(np.count_nonzero(board == 0)) == 0:
            return 0.0
        return None

    def _legal_moves(self, board: np.ndarray) -> List[int]:
        return np.flatnonzero(board.reshape(-1) == 0).astype(np.int32).tolist()

    def _rank_candidates(self, board: np.ndarray, legal_moves: Sequence[int], player: int) -> List[int]:
        frontier = _frontier_moves(board, legal_moves, self.board_size, radius=2)
        if not frontier:
            return [int(move) for move in legal_moves][: self.max_branching]

        scored: List[Tuple[float, int]] = []
        for move in frontier:
            row, col = divmod(int(move), self.board_size)
            board[row, col] = int(player)
            score = self._quick_move_score(board, row, col, int(player))
            board[row, col] = 0
            scored.append((float(score), int(move)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored[: self.max_branching]]

    def _quick_move_score(self, board: np.ndarray, row: int, col: int, player: int) -> float:
        win_length = min(5, self.board_size)
        if self._has_win(board, player=player, win_length=win_length):
            return 1_000_000.0

        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        score = 0.0
        for dr, dc in directions:
            length, open_ends = _line_length_open_ends(board, row, col, dr, dc, player)
            score += self._run_weight(length, open_ends)
        return score

    def _evaluate_board(self, board: np.ndarray) -> float:
        own = self._pattern_score(board, player=1)
        opp = self._pattern_score(board, player=-1)
        center = self._center_control(board)
        return own - 1.08 * opp + center

    def _pattern_score(self, board: np.ndarray, player: int) -> float:
        size = self.board_size
        score = 0.0
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))

        for row in range(size):
            for col in range(size):
                if int(board[row, col]) != int(player):
                    continue
                for dr, dc in directions:
                    prev_r = row - dr
                    prev_c = col - dc
                    if 0 <= prev_r < size and 0 <= prev_c < size and int(board[prev_r, prev_c]) == int(player):
                        continue

                    run = 1
                    r = row + dr
                    c = col + dc
                    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == int(player):
                        run += 1
                        r += dr
                        c += dc

                    open_ends = 0
                    if 0 <= prev_r < size and 0 <= prev_c < size and int(board[prev_r, prev_c]) == 0:
                        open_ends += 1
                    if 0 <= r < size and 0 <= c < size and int(board[r, c]) == 0:
                        open_ends += 1

                    score += self._run_weight(run, open_ends)
        return score

    def _run_weight(self, run: int, open_ends: int) -> float:
        win_length = min(5, self.board_size)
        if run >= win_length:
            return 200_000.0
        if run <= 1:
            return 0.0

        if run == win_length - 1:
            return 7_500.0 if open_ends == 2 else 2_500.0 if open_ends == 1 else 0.0
        if run == win_length - 2:
            return 900.0 if open_ends == 2 else 260.0 if open_ends == 1 else 0.0
        if run == win_length - 3:
            return 140.0 if open_ends == 2 else 35.0 if open_ends == 1 else 0.0
        return 25.0 if open_ends == 2 else 8.0 if open_ends == 1 else 0.0

    def _center_control(self, board: np.ndarray) -> float:
        center = (self.board_size - 1) / 2.0
        total = 0.0
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell = int(board[row, col])
                if cell == 0:
                    continue
                distance = np.sqrt((row - center) ** 2 + (col - center) ** 2)
                weight = 1.0 / (1.0 + distance)
                total += float(cell) * float(weight) * 22.0
        return total

    def _has_win(self, board: np.ndarray, player: int, win_length: int) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for row in range(self.board_size):
            for col in range(self.board_size):
                if int(board[row, col]) != int(player):
                    continue
                for dr, dc in directions:
                    length = 1
                    length += _count_connected(board, row, col, dr, dc, int(player))
                    length += _count_connected(board, row, col, -dr, -dc, int(player))
                    if length >= win_length:
                        return True
        return False


def create_agent(
    board_size: int,
    algorithm: str = "auto",
    learning_rate: Optional[float] = None,
    gamma: float = 0.9,
    minimax_depth: int = 3,
) -> MinimaxAgent:
    del algorithm, learning_rate, gamma
    return MinimaxAgent(board_size=board_size, max_depth=minimax_depth)
