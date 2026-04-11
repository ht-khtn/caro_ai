from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from agent import DQNAgent, MinimaxAgent, QLearningAgent, Transition, create_agent, get_torch_device_info
from environment import GomokuEnv


def _agent_backend_info(agent: Union[QLearningAgent, DQNAgent, MinimaxAgent]) -> str:
    if isinstance(agent, MinimaxAgent):
        return f"minimax:depth={agent.max_depth}"
    if isinstance(agent, DQNAgent):
        return agent.backend_info()
    return "q_learning:cpu"


def _apply_training_hyperparams(agent, epsilon_min: float, epsilon_decay: float) -> None:
    if hasattr(agent, "epsilon_min"):
        agent.epsilon_min = float(epsilon_min)
    if hasattr(agent, "epsilon_decay"):
        agent.epsilon_decay = float(epsilon_decay)


def _learn_transition(agent, pending: Dict[int, Optional[Transition]], player: int, transition: Transition, result, store_current: bool) -> None:
    if isinstance(agent, MinimaxAgent):
        return
    if isinstance(agent, QLearningAgent):
        learn_fn = agent.learn_transition
    else:
        learn_fn = agent.learn_from_transition

    if result.done and result.info.get("winner"):
        transition.reward = 100.0 if int(result.info["winner"]) == player else -100.0
        if store_current:
            learn_fn(transition)
        opponent = -player
        pending_opponent = pending.get(opponent)
        if pending_opponent is not None:
            pending_opponent.reward = -100.0 if int(result.info["winner"]) == player else pending_opponent.reward
            pending_opponent.done = True
            learn_fn(pending_opponent)
            pending[opponent] = None
        if store_current:
            pending[player] = None
        return

    pending_opponent = pending.get(-player)
    if pending_opponent is not None:
        learn_fn(pending_opponent)
        pending[-player] = None

    if store_current:
        pending[player] = transition
        if result.done:
            learn_fn(transition)
            pending[player] = None


def _save_checkpoint(agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))


def train_headless(
    board_size: int,
    algorithm: str,
    minimax_depth: int,
    episodes: int,
    opponent: str,
    save_path: Path,
    save_every: int,
    save_every_seconds: int,
    epsilon_min: float,
    epsilon_decay: float,
    report_every: int = 50,
    report_every_seconds: int = 15,
) -> None:
    env = GomokuEnv(board_size=board_size)
    agent: Union[QLearningAgent, DQNAgent, MinimaxAgent] = create_agent(
        board_size=board_size,
        algorithm=algorithm,
        minimax_depth=int(minimax_depth),
    )
    if hasattr(agent, "set_board_size"):
        agent.set_board_size(board_size)
    _apply_training_hyperparams(agent, epsilon_min, epsilon_decay)

    pending: Dict[int, Optional[Transition]] = {1: None, -1: None}
    last_save_time = time.time()
    last_report_time = time.time()
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0

    print(
        "Headless training started | board={board} | algo={algo} | opponent={opp} | episodes={ep} | backend={backend}".format(
            board=board_size,
            algo=algorithm,
            opp=opponent,
            ep=episodes,
            backend=_agent_backend_info(agent),
        )
    )
    print("Device: {device}".format(device=get_torch_device_info()))

    for episode in range(1, episodes + 1):
        env.reset()
        pending = {1: None, -1: None}
        done = False
        moves = 0

        while not done:
            current_player = int(env.current_player)
            legal_moves = env.legal_moves()
            if opponent == "random" and current_player == -1:
                action = int(np.random.choice(legal_moves)) if legal_moves else 0
            else:
                state = env.get_perspective_board(current_player)
                explore = not isinstance(agent, MinimaxAgent)
                action = int(agent.choose_action(state, legal_moves, explore=explore))

            state = env.get_perspective_board(current_player)
            result = env.step(action)
            next_state = env.get_perspective_board(-current_player) if not result.done else np.zeros_like(state)
            transition = Transition(
                state=state,
                action=int(action),
                reward=float(result.reward),
                next_state=next_state,
                done=bool(result.done),
                board_size=board_size,
            )

            store_current = opponent != "random" or current_player == 1
            if not isinstance(agent, MinimaxAgent):
                _learn_transition(agent, pending, current_player, transition, result, store_current=store_current)
            done = bool(result.done)
            moves += 1

        total_moves += moves
        if env.winner == 1:
            p1_wins += 1
        elif env.winner == -1:
            p2_wins += 1
        else:
            draws += 1

        if save_every > 0 and episode % save_every == 0:
            _save_checkpoint(agent, save_path)
            print("Checkpoint saved at episode {ep}: {path}".format(ep=episode, path=save_path))
        if save_every_seconds > 0 and (time.time() - last_save_time) >= save_every_seconds:
            _save_checkpoint(agent, save_path)
            last_save_time = time.time()
            print("Checkpoint saved (time-based) at episode {ep}: {path}".format(ep=episode, path=save_path))

        now = time.time()
        if report_every > 0 and episode % report_every == 0:
            epsilon = float(getattr(agent, "epsilon", 0.0))
            updates = int(getattr(agent, "total_updates", 0))
            avg_moves = total_moves / max(1, episode)
            print(
                "Episode {ep}/{total} | P1 {p1} P2 {p2} D {d} | avg moves {avg:.1f} | epsilon {eps:.3f} | updates {upd}".format(
                    ep=episode,
                    total=episodes,
                    p1=p1_wins,
                    p2=p2_wins,
                    d=draws,
                    avg=avg_moves,
                    eps=epsilon,
                    upd=updates,
                )
            )
            last_report_time = now
        elif report_every_seconds > 0 and (now - last_report_time) >= report_every_seconds:
            epsilon = float(getattr(agent, "epsilon", 0.0))
            updates = int(getattr(agent, "total_updates", 0))
            avg_moves = total_moves / max(1, episode)
            print(
                "Episode {ep}/{total} | P1 {p1} P2 {p2} D {d} | avg moves {avg:.1f} | epsilon {eps:.3f} | updates {upd}".format(
                    ep=episode,
                    total=episodes,
                    p1=p1_wins,
                    p2=p2_wins,
                    d=draws,
                    avg=avg_moves,
                    eps=epsilon,
                    upd=updates,
                )
            )
            last_report_time = now

    _save_checkpoint(agent, save_path)
    print("Training finished. Final checkpoint saved to {path}.".format(path=save_path))


def main() -> None:
    from gui import run_app

    run_app()


if __name__ == "__main__":
    main()
