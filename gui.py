from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import customtkinter as ctk
import numpy as np
from tkinter import Canvas, filedialog, messagebox

from agent import DQNAgent, QLearningAgent, Transition, create_agent
from environment import GomokuEnv


class GomokuApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Gomoku RL Trainer")
        self.geometry("1320x860")
        self.minsize(1200, 780)

        self.env = GomokuEnv(board_size=15)
        self.agent = create_agent(board_size=15, algorithm="auto")
        self.pending_transitions: Dict[int, Optional[Transition]] = {1: None, -1: None}

        self.mode_var = ctk.StringVar(value="Training")
        self.training_opponent_var = ctk.StringVar(value="Self-Play")
        self.algorithm_var = ctk.StringVar(value="Auto")
        self.human_side_var = ctk.StringVar(value="First")
        self.board_size_var = ctk.IntVar(value=15)
        self.board_size_display_var = ctk.StringVar(value="15 x 15")
        self.status_var = ctk.StringVar(value="Ready")
        self.move_delay_var = ctk.DoubleVar(value=350)
        self.stats_var = ctk.StringVar(value="Episodes: 0 | Moves: 0 | Epsilon: 1.00")
        self.score_var = ctk.StringVar(value="P1 Wins: 0 | P2 Wins: 0 | Draws: 0")
        self.turn_var = ctk.StringVar(value="Current turn: Black")

        self.is_running = False
        self.game_over = False
        self.total_episodes = 0
        self.total_moves = 0
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        self.ai_wins = 0
        self.ai_losses = 0
        self.human_wins = 0
        self.human_losses = 0
        self.last_move: Optional[tuple[int, int]] = None
        self.board_pixel_size = 720

        self._build_layout()
        self._bind_events()
        self._refresh_board()
        self._update_statistics()

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=330, corner_radius=18)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.sidebar.grid_propagate(False)

        self.main = ctk.CTkFrame(self, corner_radius=18)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(10, 18), pady=18)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="Gomoku RL Trainer", font=ctk.CTkFont(size=26, weight="bold"))
        title.pack(padx=18, pady=(18, 8), anchor="w")

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Self-play, human vs AI, curriculum board sizes, and live learning.",
            wraplength=280,
            justify="left",
            text_color="#B8C4D6",
        )
        subtitle.pack(padx=18, pady=(0, 16), anchor="w")

        self._add_section_label(self.sidebar, "Game Setup")
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, values=["Training", "Human vs AI"], variable=self.mode_var, command=self._on_mode_change)
        self.mode_menu.pack(fill="x", padx=18, pady=(0, 10))

        self.opponent_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Self-Play", "Random Bot"],
            variable=self.training_opponent_var,
        )
        self.opponent_menu.pack(fill="x", padx=18, pady=(0, 10))

        self.human_side_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["First", "Second"],
            variable=self.human_side_var,
            command=lambda _value: self._refresh_turn_label(),
        )
        self.human_side_menu.pack(fill="x", padx=18, pady=(0, 14))

        self._add_section_label(self.sidebar, "Algorithm")
        self.algorithm_menu = ctk.CTkOptionMenu(self.sidebar, values=["Auto", "Q-Learning", "DQN"], variable=self.algorithm_var)
        self.algorithm_menu.pack(fill="x", padx=18, pady=(0, 14))

        self._add_section_label(self.sidebar, "Curriculum Board Size")
        self.board_slider = ctk.CTkSlider(self.sidebar, from_=3, to=15, number_of_steps=12, command=self._on_board_slider)
        self.board_slider.set(15)
        self.board_slider.pack(fill="x", padx=18, pady=(0, 6))
        self.board_size_label = ctk.CTkLabel(self.sidebar, textvariable=self.board_size_display_var, font=ctk.CTkFont(size=14, weight="bold"))
        self.board_size_label.pack(padx=18, pady=(0, 8), anchor="w")
        self.apply_board_button = ctk.CTkButton(self.sidebar, text="Apply Board Size", command=self._apply_board_size)
        self.apply_board_button.pack(fill="x", padx=18, pady=(0, 14))

        self._add_section_label(self.sidebar, "Training Speed")
        self.speed_slider = ctk.CTkSlider(self.sidebar, from_=0, to=2000, number_of_steps=40, command=self._on_speed_change)
        self.speed_slider.set(350)
        self.speed_slider.pack(fill="x", padx=18, pady=(0, 6))
        self.speed_value_label = ctk.CTkLabel(self.sidebar, text="350 ms between AI moves", text_color="#B8C4D6")
        self.speed_value_label.pack(padx=18, pady=(0, 14), anchor="w")

        self._add_section_label(self.sidebar, "Controls")
        controls_row_1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        controls_row_1.pack(fill="x", padx=18, pady=(0, 8))
        self.start_button = ctk.CTkButton(controls_row_1, text="Start", command=self._toggle_running)
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.reset_button = ctk.CTkButton(controls_row_1, text="Reset", command=self._reset_game)
        self.reset_button.pack(side="left", expand=True, fill="x", padx=(6, 0))

        controls_row_2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        controls_row_2.pack(fill="x", padx=18, pady=(0, 8))
        self.save_button = ctk.CTkButton(controls_row_2, text="Save Model", command=self._save_model)
        self.save_button.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.load_button = ctk.CTkButton(controls_row_2, text="Load Model", command=self._load_model)
        self.load_button.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self._add_section_label(self.sidebar, "Live Status")
        self.status_label = ctk.CTkLabel(self.sidebar, textvariable=self.status_var, wraplength=290, justify="left", text_color="#DCE7F5")
        self.status_label.pack(fill="x", padx=18, pady=(0, 8), anchor="w")
        self.turn_label = ctk.CTkLabel(self.sidebar, textvariable=self.turn_var, wraplength=290, justify="left", text_color="#8AD4FF")
        self.turn_label.pack(fill="x", padx=18, pady=(0, 8), anchor="w")

        self.stats_label = ctk.CTkLabel(self.sidebar, textvariable=self.stats_var, wraplength=290, justify="left", text_color="#B8C4D6")
        self.stats_label.pack(fill="x", padx=18, pady=(0, 2), anchor="w")
        self.score_label = ctk.CTkLabel(self.sidebar, textvariable=self.score_var, wraplength=290, justify="left", text_color="#B8C4D6")
        self.score_label.pack(fill="x", padx=18, pady=(0, 18), anchor="w")

        self.board_container = ctk.CTkFrame(self.main, corner_radius=18)
        self.board_container.grid(row=1, column=0, sticky="nsew", padx=18, pady=18)
        self.board_container.grid_propagate(False)

        self.canvas = Canvas(
            self.board_container,
            width=self.board_pixel_size,
            height=self.board_pixel_size,
            highlightthickness=0,
            bg="#D7B57D",
        )
        self.canvas.pack(fill="both", expand=True, padx=16, pady=16)

        footer = ctk.CTkLabel(
            self.main,
            text="Black = Player 1, White = Player 2. Last move is highlighted with an orange ring.",
            text_color="#B8C4D6",
        )
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))

    def _add_section_label(self, parent: ctk.CTkFrame, text: str) -> None:
        label = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=15, weight="bold"), anchor="w")
        label.pack(fill="x", padx=18, pady=(8, 6), anchor="w")

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.is_running = False
        self.destroy()

    def _on_board_slider(self, value: float) -> None:
        size = int(round(float(value)))
        self.board_size_var.set(size)
        self.board_size_display_var.set(f"{size} x {size}")

    def _on_speed_change(self, value: float) -> None:
        delay_ms = int(round(float(value)))
        self.move_delay_var.set(delay_ms)
        self.speed_value_label.configure(text=f"{delay_ms} ms between AI moves")

    def _on_mode_change(self, value: str) -> None:
        self.mode_var.set(value)
        self._refresh_turn_label()
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        if self.mode_var.get() == "Training":
            self.opponent_menu.configure(state="normal")
            self.human_side_menu.configure(state="disabled")
            self.start_button.configure(text="Pause" if self.is_running else "Start")
        else:
            self.opponent_menu.configure(state="disabled")
            self.human_side_menu.configure(state="normal")
            self.start_button.configure(text="Pause" if self.is_running else "Start")

    def _apply_board_size(self) -> None:
        size = int(self.board_size_var.get())
        algorithm = self.algorithm_var.get().lower().strip()
        if algorithm == "q-learning":
            algorithm = "q_learning"
        self.env = GomokuEnv(board_size=size)
        self.agent = create_agent(board_size=size, algorithm=algorithm)
        self.pending_transitions = {1: None, -1: None}
        self.total_episodes = 0
        self.total_moves = 0
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        self.ai_wins = 0
        self.ai_losses = 0
        self.human_wins = 0
        self.human_losses = 0
        self.last_move = None
        self.game_over = False
        self.is_running = False
        self.start_button.configure(text="Start")
        self.status_var.set(f"Applied {size} x {size} board. New model initialized from scratch.")
        self._refresh_turn_label()
        self._update_statistics()
        self._refresh_board()

    def _toggle_running(self) -> None:
        self.is_running = not self.is_running
        self.start_button.configure(text="Pause" if self.is_running else "Start")
        self.status_var.set("Training running." if self.is_running else "Paused.")
        if self.is_running:
            self._schedule_next_step(1)

    def _reset_game(self) -> None:
        self.env.reset()
        self.pending_transitions = {1: None, -1: None}
        self.game_over = False
        self.last_move = None
        self.status_var.set("Board reset.")
        self._refresh_turn_label()
        self._refresh_board()
        if self.mode_var.get() == "Human vs AI" and self._is_ai_turn():
            self._schedule_next_step(250)

    def _schedule_next_step(self, delay_ms: int) -> None:
        self.after(max(1, delay_ms), self._training_or_play_step)

    def _training_or_play_step(self) -> None:
        if not self.is_running:
            return
        if self.mode_var.get() == "Training":
            self._training_step()
        else:
            self._auto_ai_step()

    def _training_step(self) -> None:
        if self.game_over:
            self._finish_episode()
            self._schedule_next_step(600)
            return

        current_player = int(self.env.current_player)
        if self.training_opponent_var.get() == "Random Bot" and current_player == -1:
            action = self._choose_random_move()
        else:
            action = self._choose_agent_move(current_player)

        self._apply_move(action, current_player, learning_enabled=True)
        self._refresh_board()

        if self.game_over:
            self._finish_episode()
            self._schedule_next_step(600)
        else:
            self._schedule_next_step(int(self.move_delay_var.get()))

    def _auto_ai_step(self) -> None:
        if self.game_over:
            return
        if not self._is_ai_turn():
            return
        current_player = int(self.env.current_player)
        action = self._choose_agent_move(current_player)
        self._apply_move(action, current_player, learning_enabled=True)
        self._refresh_board()

        if self.game_over:
            self._finish_episode()
        elif self.is_running:
            self._schedule_next_step(int(self.move_delay_var.get()))

    def _is_ai_turn(self) -> bool:
        if self.mode_var.get() == "Training":
            return True
        human_player = 1 if self.human_side_var.get() == "First" else -1
        return int(self.env.current_player) != human_player

    def _choose_random_move(self) -> int:
        legal = self.env.legal_moves()
        if not legal:
            return 0
        return int(np.random.choice(legal))

    def _choose_agent_move(self, player: int) -> int:
        state = self.env.get_perspective_board(player)
        legal_moves = self.env.legal_moves()
        return int(self.agent.choose_action(state, legal_moves, explore=True))

    def _on_canvas_click(self, event) -> None:
        if self.mode_var.get() != "Human vs AI":
            return
        if self.game_over:
            return
        human_player = 1 if self.human_side_var.get() == "First" else -1
        if int(self.env.current_player) != human_player:
            return

        row, col = self._pixel_to_move(event.x, event.y)
        if row is None or col is None:
            return
        self._apply_move(row * self.env.board_size + col, human_player, learning_enabled=False)
        self._refresh_board()
        if self.game_over:
            self._finish_episode()
        elif self._is_ai_turn():
            self._schedule_next_step(int(self.move_delay_var.get()))
        else:
            self._refresh_turn_label()

    def _pixel_to_move(self, x: int, y: int) -> tuple[Optional[int], Optional[int]]:
        size = self.env.board_size
        margin = 28
        board_length = self.board_pixel_size - margin * 2
        cell = board_length / size
        col = int((x - margin) / cell)
        row = int((y - margin) / cell)
        if not (0 <= row < size and 0 <= col < size):
            return None, None
        return row, col

    def _apply_move(self, action: int, player: int, learning_enabled: bool) -> None:
        state = self.env.get_perspective_board(player)
        result = self.env.step(action)
        next_state = self.env.get_perspective_board(-player) if not result.done else np.zeros_like(state)
        transition = Transition(state=state, action=int(action), reward=float(result.reward), next_state=next_state, done=bool(result.done))

        ai_player = 1 if self.human_side_var.get() == "Second" else -1
        if self.mode_var.get() == "Training":
            store_current = self.training_opponent_var.get() != "Random Bot" or player == 1
        else:
            store_current = player == ai_player
        if learning_enabled or self.mode_var.get() == "Human vs AI":
            self._process_learning(player, transition, result, store_current=store_current)

        self.total_moves += 1
        self.last_move = result.info.get("move") if isinstance(result.info, dict) else None
        self.game_over = bool(result.done)

        if result.info.get("illegal"):
            self.status_var.set("Illegal move: -50 and turn skipped.")
        elif result.info.get("winner"):
            winner = int(result.info["winner"])
            self.status_var.set("Player 1 wins." if winner == 1 else "Player 2 wins.")
        elif result.done:
            self.status_var.set("Draw: board is full.")
        else:
            move_name = "Black" if self.env.current_player == -1 else "White"
            self.status_var.set(f"Move played. Next turn: {move_name}.")

        self._refresh_turn_label()
        self._update_statistics()
        if self.mode_var.get() == "Human vs AI" and not self.game_over and self._is_ai_turn():
            self.after(int(self.move_delay_var.get()), self._auto_ai_step)

    def _process_learning(self, player: int, transition: Transition, result, store_current: bool) -> None:
        if isinstance(self.agent, QLearningAgent):
            self._learn_transition_q(player, transition, result, store_current)
        else:
            self._learn_transition_dqn(player, transition, result, store_current)

    def _learn_transition_q(self, player: int, transition: Transition, result, store_current: bool) -> None:
        if result.done and result.info.get("winner"):
            transition.reward = 100.0 if int(result.info["winner"]) == player else -100.0
            if store_current:
                self.agent.learn_transition(transition)
            opponent = -player
            pending = self.pending_transitions.get(opponent)
            if pending is not None:
                pending.reward = -100.0 if int(result.info["winner"]) == player else pending.reward
                pending.done = True
                self.agent.learn_transition(pending)
                self.pending_transitions[opponent] = None
            if store_current:
                self.pending_transitions[player] = None
            return

        pending = self.pending_transitions.get(-player)
        if pending is not None:
            self.agent.learn_transition(pending)
            self.pending_transitions[-player] = None

        if store_current:
            self.pending_transitions[player] = transition
            if result.done:
                self.agent.learn_transition(transition)
                self.pending_transitions[player] = None

    def _learn_transition_dqn(self, player: int, transition: Transition, result, store_current: bool) -> None:
        if result.done and result.info.get("winner"):
            transition.reward = 100.0 if int(result.info["winner"]) == player else -100.0
            if store_current:
                self.agent.learn_from_transition(transition)
            opponent = -player
            pending = self.pending_transitions.get(opponent)
            if pending is not None:
                pending.reward = -100.0 if int(result.info["winner"]) == player else pending.reward
                pending.done = True
                self.agent.learn_from_transition(pending)
                self.pending_transitions[opponent] = None
            if store_current:
                self.pending_transitions[player] = None
            return

        pending = self.pending_transitions.get(-player)
        if pending is not None:
            self.agent.learn_from_transition(pending)
            self.pending_transitions[-player] = None

        if store_current:
            self.pending_transitions[player] = transition
            if result.done:
                self.agent.learn_from_transition(transition)
                self.pending_transitions[player] = None

    def _finish_episode(self) -> None:
        if self.env.winner == 1:
            self.p1_wins += 1
            if self.mode_var.get() == "Human vs AI":
                ai_player = 1 if self.human_side_var.get() == "Second" else -1
                if ai_player == 1:
                    self.ai_wins += 1
                    self.human_losses += 1
                else:
                    self.human_wins += 1
                    self.ai_losses += 1
        elif self.env.winner == -1:
            self.p2_wins += 1
            if self.mode_var.get() == "Human vs AI":
                ai_player = 1 if self.human_side_var.get() == "Second" else -1
                if ai_player == -1:
                    self.ai_wins += 1
                    self.human_losses += 1
                else:
                    self.human_wins += 1
                    self.ai_losses += 1
        else:
            self.draws += 1

        self.total_episodes += 1
        self._update_statistics()
        self.pending_transitions = {1: None, -1: None}
        self.game_over = False
        self.env.reset()
        self._refresh_board()
        self._refresh_turn_label()
        if self.mode_var.get() == "Human vs AI" and self._is_ai_turn():
            self._schedule_next_step(250)

    def _refresh_turn_label(self) -> None:
        if self.mode_var.get() == "Human vs AI":
            human_player = 1 if self.human_side_var.get() == "First" else -1
            turn = "Human" if int(self.env.current_player) == human_player else "AI"
            color = "Black" if self.env.current_player == 1 else "White"
            self.turn_var.set(f"Current turn: {turn} ({color})")
        else:
            color = "Black" if self.env.current_player == 1 else "White"
            self.turn_var.set(f"Current turn: {color}")

    def _update_statistics(self) -> None:
        epsilon = getattr(self.agent, "epsilon", 1.0)
        updates = getattr(self.agent, "total_updates", 0)
        self.stats_var.set(f"Episodes: {self.total_episodes} | Moves: {self.total_moves} | Updates: {updates} | Epsilon: {epsilon:.3f}")
        if self.mode_var.get() == "Human vs AI":
            self.score_var.set(f"AI Wins: {self.ai_wins} | AI Losses: {self.ai_losses} | Draws: {self.draws}")
        else:
            self.score_var.set(f"P1 Wins: {self.p1_wins} | P2 Wins: {self.p2_wins} | Draws: {self.draws}")

    def _refresh_board(self) -> None:
        self.canvas.delete("all")
        size = self.env.board_size
        margin = 28
        board_length = self.board_pixel_size - margin * 2
        cell = board_length / size

        for index in range(size + 1):
            x = margin + index * cell
            self.canvas.create_line(margin, x, margin + board_length, x, fill="#4A3A1F", width=1)
            self.canvas.create_line(x, margin, x, margin + board_length, fill="#4A3A1F", width=1)

        self._draw_labels(size, margin, cell)
        self._draw_star_points(size, margin, cell)

        radius = cell * 0.36
        for row in range(size):
            for col in range(size):
                value = int(self.env.board[row, col])
                if value == 0:
                    continue
                cx = margin + col * cell + cell / 2
                cy = margin + row * cell + cell / 2
                fill = "#111111" if value == 1 else "#F2F2F2"
                outline = "#000000" if value == 1 else "#B8B8B8"
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, outline=outline, width=2)

        if self.env.last_move is not None:
            row, col = self.env.last_move
            cx = margin + col * cell + cell / 2
            cy = margin + row * cell + cell / 2
            self.canvas.create_oval(cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6, outline="#FF9F43", width=3)

        self.canvas.configure(bg="#D7B57D")

    def _draw_labels(self, size: int, margin: int, cell: float) -> None:
        for index in range(size):
            label = str(index + 1)
            x = margin + index * cell + cell / 2
            y = margin / 2
            self.canvas.create_text(x, y, text=label, fill="#3A2C12", font=("Segoe UI", 9, "bold"))
            self.canvas.create_text(margin / 2, margin + index * cell + cell / 2, text=label, fill="#3A2C12", font=("Segoe UI", 9, "bold"))

    def _draw_star_points(self, size: int, margin: int, cell: float) -> None:
        if size < 9:
            return
        points = [size // 4, size // 2, (3 * size) // 4]
        radius = max(2, int(cell * 0.08))
        for row in points:
            for col in points:
                if 0 <= row < size and 0 <= col < size:
                    cx = margin + col * cell + cell / 2
                    cy = margin + row * cell + cell / 2
                    self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#3A2C12", outline="")

    def _save_model(self) -> None:
        file_path = filedialog.asksaveasfilename(
            title="Save Gomoku model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self.agent.save(file_path)
            self.status_var.set(f"Model saved to {file_path}")
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Save failed", str(exc))

    def _load_model(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Load Gomoku model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            loaded = None
            with open(file_path, "rb") as handle:
                payload = pickle.load(handle)
            model_type = payload.get("type")
            board_size = int(payload.get("board_size", self.env.board_size))
            if model_type == "q_learning":
                loaded = QLearningAgent.load(file_path)
            elif model_type == "dqn":
                loaded = DQNAgent.load(file_path)
            else:
                raise ValueError("Unsupported model file")

            self.env = GomokuEnv(board_size=board_size)
            self.agent = loaded
            self.board_size_var.set(board_size)
            self.board_size_display_var.set(f"{board_size} x {board_size}")
            self.board_slider.set(board_size)
            self.pending_transitions = {1: None, -1: None}
            self.game_over = False
            self.is_running = False
            self.start_button.configure(text="Start")
            self.status_var.set(f"Loaded model from {os.path.basename(file_path)}")
            self._refresh_board()
            self._refresh_turn_label()
            self._update_statistics()
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Load failed", str(exc))


def run_app() -> None:
    app = GomokuApp()
    app.mainloop()
