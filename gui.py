from __future__ import annotations

import os
import pickle
from typing import Dict, Optional

import customtkinter as ctk
import numpy as np
from tkinter import Canvas, Label, Toplevel, filedialog, messagebox

from agent import DQNAgent, MinimaxAgent, QLearningAgent, Transition, create_agent
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
        self.move_delay_var = ctk.DoubleVar(value=140)
        self.no_ui_var = ctk.BooleanVar(value=False)
        self.stats_var = ctk.StringVar(value="Episodes: 0 | Moves: 0 | Epsilon: 1.00")
        self.score_var = ctk.StringVar(value="P1 Wins: 0 | P2 Wins: 0 | Draws: 0")
        self.turn_var = ctk.StringVar(value="Current turn: X")
        self.x_score_var = ctk.StringVar(value="0")
        self.o_score_var = ctk.StringVar(value="0")
        self.ai_level_var = ctk.StringVar(value="0/100")
        self.center_focus_var = ctk.StringVar(value="Center focus: 0%")

        self.is_running = False
        self.game_over = False
        self.finish_scheduled = False
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
        self.win_line_cells: Optional[list[tuple[int, int]]] = None
        self.show_win_line = False
        self.win_line_delay_ms = 480
        self.center_opening_hits = 0
        self.center_opening_samples = 0
        self.board_pixel_size = 720
        self._ui_tick_counter = 0
        self._tooltip_window: Optional[Toplevel] = None
        self._tooltip_after_id: Optional[str] = None
        self._tooltip_message = ""

        self._build_layout()
        self._bind_events()
        self._refresh_board()
        self._update_statistics()

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=18)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self.main = ctk.CTkFrame(self, corner_radius=18)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(10, 18), pady=18)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_columnconfigure(1, weight=0)
        self.main.grid_rowconfigure(0, weight=0)
        self.main.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="Gomoku RL Trainer", font=ctk.CTkFont(size=26, weight="bold"))
        title.grid(row=0, column=0, padx=18, pady=(18, 8), sticky="w")

        self.settings_scroll = ctk.CTkScrollableFrame(
            self.sidebar,
            width=308,
            height=740,
            corner_radius=14,
            fg_color=("#E9ECF2", "#1C2430"),
        )
        self.settings_scroll.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        subtitle = ctk.CTkLabel(
            self.settings_scroll,
            text="Self-play, human vs AI, curriculum board sizes, and live learning.",
            wraplength=280,
            justify="left",
            text_color="#B8C4D6",
        )
        subtitle.pack(padx=14, pady=(10, 16), anchor="w")

        self._add_section_label(self.settings_scroll, "Game Setup", "Chon che do train tu dau hoac dau voi AI, va doi vai tro nguoi choi.")
        self.mode_menu = ctk.CTkOptionMenu(self.settings_scroll, values=["Training", "Human vs AI"], variable=self.mode_var, command=self._on_mode_change)
        self.mode_menu.pack(fill="x", padx=14, pady=(0, 10))

        self.opponent_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            values=["Self-Play", "Random Bot"],
            variable=self.training_opponent_var,
        )
        self.opponent_menu.pack(fill="x", padx=14, pady=(0, 10))

        self.human_side_menu = ctk.CTkOptionMenu(
            self.settings_scroll,
            values=["First", "Second"],
            variable=self.human_side_var,
            command=lambda _value: self._refresh_turn_label(),
        )
        self.human_side_menu.pack(fill="x", padx=14, pady=(0, 14))

        self._add_section_label(self.settings_scroll, "Algorithm", "Auto mac dinh dung Minimax + Alpha-Beta. Van co the chon Q-Learning hoac DQN.")
        self.algorithm_menu = ctk.CTkOptionMenu(self.settings_scroll, values=["Auto", "Minimax", "Q-Learning", "DQN"], variable=self.algorithm_var)
        self.algorithm_menu.pack(fill="x", padx=14, pady=(0, 14))

        self._add_section_label(self.settings_scroll, "Curriculum Board Size", "Bat dau ban nho de hoc khai niem, sau do tang dan toi 15x15.")
        self.board_slider = ctk.CTkSlider(self.settings_scroll, from_=3, to=15, number_of_steps=12, command=self._on_board_slider)
        self.board_slider.set(15)
        self.board_slider.pack(fill="x", padx=14, pady=(0, 6))
        self.board_size_label = ctk.CTkLabel(self.settings_scroll, textvariable=self.board_size_display_var, font=ctk.CTkFont(size=14, weight="bold"))
        self.board_size_label.pack(padx=14, pady=(0, 8), anchor="w")
        self.apply_board_button = ctk.CTkButton(self.settings_scroll, text="Apply Board Size", command=self._apply_board_size)
        self.apply_board_button.pack(fill="x", padx=14, pady=(0, 14))

        self._add_section_label(self.settings_scroll, "Training Speed", "Do tre giua cac nuoc AI. Tang de quan sat, giam de train nhanh.")
        self.speed_slider = ctk.CTkSlider(self.settings_scroll, from_=0, to=2000, number_of_steps=40, command=self._on_speed_change)
        self.speed_slider.set(140)
        self.speed_slider.pack(fill="x", padx=14, pady=(0, 6))
        self.speed_value_label = ctk.CTkLabel(self.settings_scroll, text="140 ms between AI moves", text_color="#B8C4D6")
        self.speed_value_label.pack(padx=14, pady=(0, 6), anchor="w")
        self.no_ui_checkbox = ctk.CTkCheckBox(
            self.settings_scroll,
            text="No UI (train faster)",
            variable=self.no_ui_var,
            command=self._on_no_ui_toggle,
        )
        self.no_ui_checkbox.pack(padx=14, pady=(0, 14), anchor="w")

        toolbar = ctk.CTkFrame(self.main, corner_radius=14, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(10, 6))
        for col in range(5):
            toolbar.grid_columnconfigure(col, weight=1)

        button_style = dict(width=42, height=34, fg_color="transparent", hover_color="#263243", text_color="#EAF2FF")
        self.start_button = ctk.CTkButton(toolbar, text="▶", command=self._toggle_running, **button_style)
        self.start_button.grid(row=0, column=1, padx=3, pady=4)
        self.reset_button = ctk.CTkButton(toolbar, text="↺", command=self._reset_game, **button_style)
        self.reset_button.grid(row=0, column=2, padx=3, pady=4)
        self.save_button = ctk.CTkButton(toolbar, text="💾", command=self._save_model, **button_style)
        self.save_button.grid(row=0, column=3, padx=3, pady=4)
        self.load_button = ctk.CTkButton(toolbar, text="📂", command=self._load_model, **button_style)
        self.load_button.grid(row=0, column=4, padx=3, pady=4)
        self.mode_hint = ctk.CTkLabel(toolbar, text="Controls", text_color="#9AB2D9")
        self.mode_hint.grid(row=0, column=0, sticky="w", padx=(10, 6))

        self.board_container = ctk.CTkFrame(self.main, corner_radius=18)
        self.board_container.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.board_container.grid_propagate(False)

        self.right_panel = ctk.CTkFrame(self.main, width=250, corner_radius=18)
        self.right_panel.grid(row=1, column=1, sticky="ns", padx=(10, 18), pady=18)
        self.right_panel.grid_propagate(False)

        live_title = ctk.CTkLabel(self.right_panel, text="Live Status", font=ctk.CTkFont(size=20, weight="bold"))
        live_title.pack(padx=14, pady=(16, 8), anchor="w")
        self.status_label = ctk.CTkLabel(self.right_panel, textvariable=self.status_var, wraplength=210, justify="left", text_color="#DCE7F5")
        self.status_label.pack(fill="x", padx=14, pady=(0, 8), anchor="w")
        self.turn_label = ctk.CTkLabel(self.right_panel, textvariable=self.turn_var, wraplength=210, justify="left", text_color="#8AD4FF")
        self.turn_label.pack(fill="x", padx=14, pady=(0, 8), anchor="w")
        self.stats_label = ctk.CTkLabel(self.right_panel, textvariable=self.stats_var, wraplength=210, justify="left", text_color="#B8C4D6")
        self.stats_label.pack(fill="x", padx=14, pady=(0, 10), anchor="w")

        self.canvas = Canvas(
            self.board_container,
            width=self.board_pixel_size,
            height=self.board_pixel_size,
            highlightthickness=0,
            bg="#D7B57D",
        )
        self.canvas.pack(fill="both", expand=True, padx=16, pady=16)

        score_title = ctk.CTkLabel(self.right_panel, text="Match Score", font=ctk.CTkFont(size=20, weight="bold"))
        score_title.pack(padx=14, pady=(8, 10), anchor="w")
        score_row = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        score_row.pack(fill="x", padx=14, pady=(0, 14))
        x_box = ctk.CTkFrame(score_row, fg_color="transparent")
        x_box.pack(side="left", expand=True, fill="x", padx=(0, 8))
        o_box = ctk.CTkFrame(score_row, fg_color="transparent")
        o_box.pack(side="left", expand=True, fill="x", padx=(8, 0))
        ctk.CTkLabel(x_box, text="X", text_color="#E74C3C", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="center")
        ctk.CTkLabel(x_box, textvariable=self.x_score_var, text_color="#E74C3C", font=ctk.CTkFont(size=30, weight="bold")).pack(anchor="center")
        ctk.CTkLabel(o_box, text="O", text_color="#3498DB", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="center")
        ctk.CTkLabel(o_box, textvariable=self.o_score_var, text_color="#3498DB", font=ctk.CTkFont(size=30, weight="bold")).pack(anchor="center")

        train_title = ctk.CTkLabel(self.right_panel, text="AI Training", font=ctk.CTkFont(size=20, weight="bold"))
        train_title.pack(padx=14, pady=(6, 10), anchor="w")
        train_level = ctk.CTkLabel(self.right_panel, textvariable=self.ai_level_var, text_color="#A2F59C", font=ctk.CTkFont(size=16, weight="bold"))
        train_level.pack(padx=14, pady=(0, 6), anchor="w")
        center_focus = ctk.CTkLabel(self.right_panel, textvariable=self.center_focus_var, text_color="#86D3FF", wraplength=220, justify="left")
        center_focus.pack(padx=14, pady=(0, 12), anchor="w")

        self.reset_ai_button = ctk.CTkButton(
            self.right_panel,
            text="Reset AI Knowledge",
            fg_color="#7A1F1F",
            hover_color="#932A2A",
            command=self._reset_ai_knowledge,
        )
        self.reset_ai_button.pack(fill="x", padx=14, pady=(6, 12))

        footer = ctk.CTkLabel(
            self.main,
            text="X do va O xanh. Nuoc vua danh duoc highlight, va khi thang se ve duong noi 5 quan.",
            text_color="#B8C4D6",
        )
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 14))

    def _add_section_label(self, parent: ctk.CTkFrame, text: str, tip: str) -> None:
        label = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=15, weight="bold"), anchor="w")
        label.pack(fill="x", padx=14, pady=(8, 6), anchor="w")
        label.bind("<Enter>", lambda event, message=tip: self._schedule_tooltip(event, message))
        label.bind("<Leave>", lambda _event: self._hide_tooltip())

    def _bind_events(self) -> None:
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.is_running = False
        self._hide_tooltip()
        self.destroy()

    def _schedule_tooltip(self, event, message: str) -> None:
        self._tooltip_message = message
        self._hide_tooltip()
        self._tooltip_after_id = self.after(800, lambda: self._show_tooltip(event))

    def _show_tooltip(self, event) -> None:
        if not self._tooltip_message:
            return
        self._tooltip_window = Toplevel(self)
        self._tooltip_window.wm_overrideredirect(True)
        x = event.widget.winfo_rootx() + 12
        y = event.widget.winfo_rooty() + 26
        self._tooltip_window.wm_geometry(f"+{x}+{y}")
        label = Label(
            self._tooltip_window,
            text=self._tooltip_message,
            justify="left",
            bg="#111823",
            fg="#EAF2FF",
            padx=10,
            pady=6,
            font=("Segoe UI", 9),
        )
        label.pack()

    def _hide_tooltip(self) -> None:
        if self._tooltip_after_id is not None:
            self.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_window is not None:
            self._tooltip_window.destroy()
            self._tooltip_window = None

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

    def _on_no_ui_toggle(self) -> None:
        if self.mode_var.get() != "Training":
            self.no_ui_var.set(False)
            self.status_var.set("No UI is available only in Training mode.")
            return
        if self.no_ui_var.get():
            self.status_var.set("No UI enabled: rendering reduced for faster training.")
        else:
            self.status_var.set("No UI disabled: normal rendering restored.")
            self._refresh_board()
            self._refresh_turn_label()
            self._update_statistics()

    def _is_no_ui_active(self) -> bool:
        return self.mode_var.get() == "Training" and bool(self.no_ui_var.get())

    def _refresh_controls(self) -> None:
        if self.mode_var.get() == "Training":
            self.opponent_menu.configure(state="normal")
            self.human_side_menu.configure(state="disabled")
            self.no_ui_checkbox.configure(state="normal")
            self.mode_hint.configure(text="Training controls")
        else:
            self.opponent_menu.configure(state="disabled")
            self.human_side_menu.configure(state="normal")
            self.no_ui_var.set(False)
            self.no_ui_checkbox.configure(state="disabled")
            self.mode_hint.configure(text="Human vs AI controls")
        self._refresh_start_icon()

    def _refresh_start_icon(self) -> None:
        self.start_button.configure(text="⏸" if self.is_running else "▶")

    def _selected_algorithm_key(self) -> str:
        selected = self.algorithm_var.get().strip().lower()
        if selected in {"q-learning", "q_learning", "q"}:
            return "q_learning"
        if selected == "dqn":
            return "dqn"
        if selected in {"minimax", "alpha-beta", "alpha_beta"}:
            return "minimax"
        return "auto"

    def _current_algorithm_key(self) -> str:
        if isinstance(self.agent, QLearningAgent):
            return "q_learning"
        if isinstance(self.agent, DQNAgent):
            return "dqn"
        if isinstance(self.agent, MinimaxAgent):
            return "minimax"
        return "auto"

    def _ensure_selected_algorithm(self) -> None:
        selected = self._selected_algorithm_key()
        current = self._current_algorithm_key()
        if selected == "auto" and current == "minimax":
            return
        if selected == current:
            return
        self.agent = create_agent(board_size=self.env.board_size, algorithm=selected)
        self.pending_transitions = {1: None, -1: None}
        self.status_var.set(f"Switched AI algorithm to {selected}.")

    def _apply_board_size(self) -> None:
        size = int(self.board_size_var.get())
        self.env = GomokuEnv(board_size=size)
        self._ensure_selected_algorithm()
        if hasattr(self.agent, "set_board_size"):
            self.agent.set_board_size(size)
        self.pending_transitions = {1: None, -1: None}
        self.last_move = None
        self.win_line_cells = None
        self.show_win_line = False
        self.game_over = False
        self.finish_scheduled = False
        self.is_running = False
        self._refresh_start_icon()
        self.status_var.set(f"Applied {size} x {size} board. Learned knowledge preserved.")
        self._refresh_turn_label()
        self._update_statistics()
        self._refresh_board()

    def _toggle_running(self) -> None:
        if not self.is_running:
            self._ensure_selected_algorithm()
        self.is_running = not self.is_running
        self._refresh_start_icon()
        self.status_var.set("Training running." if self.is_running else "Paused.")
        if self.is_running:
            self._schedule_next_step(1)

    def _reset_game(self) -> None:
        self.env.reset()
        self._ui_tick_counter = 0
        self.pending_transitions = {1: None, -1: None}
        self.game_over = False
        self.finish_scheduled = False
        self.win_line_cells = None
        self.show_win_line = False
        self.last_move = None
        self.status_var.set("Board reset.")
        self._refresh_turn_label()
        self._refresh_board()
        if self.mode_var.get() == "Human vs AI" and self._is_ai_turn():
            self.after(250, self._auto_ai_step)

    def _reset_ai_knowledge(self) -> None:
        answer = messagebox.askyesno(
            "Reset AI",
            "Reset toàn bộ tri thức AI hiện tại?\nHành động này sẽ xóa kiến thức đã học trong bộ nhớ.",
        )
        if not answer:
            return

        self.is_running = False
        self._refresh_start_icon()
        self.pending_transitions = {1: None, -1: None}
        self.center_opening_hits = 0
        self.center_opening_samples = 0

        if hasattr(self.agent, "reset_knowledge"):
            self.agent.reset_knowledge()
        else:
            if isinstance(self.agent, QLearningAgent):
                self.agent = create_agent(self.env.board_size, algorithm="q_learning")
            elif isinstance(self.agent, MinimaxAgent):
                self.agent = create_agent(self.env.board_size, algorithm="minimax")
            else:
                self.agent = create_agent(self.env.board_size, algorithm="dqn")

        if hasattr(self.agent, "set_board_size"):
            self.agent.set_board_size(self.env.board_size)

        self.status_var.set("AI knowledge reset complete.")
        self._update_statistics()

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
            return

        current_player = int(self.env.current_player)
        if self.training_opponent_var.get() == "Random Bot" and current_player == -1:
            action = self._choose_random_move()
        else:
            action = self._choose_agent_move(current_player)

        self._apply_move(action, current_player, learning_enabled=True)
        if not self._is_no_ui_active():
            self._refresh_board()

        if not self.game_over:
            self._schedule_next_step(int(self.move_delay_var.get()))

    def _auto_ai_step(self) -> None:
        if self.game_over:
            return
        if not self._is_ai_turn():
            return
        current_player = int(self.env.current_player)
        action = self._choose_agent_move(current_player)
        self._apply_move(action, current_player, learning_enabled=True)
        if not self._is_no_ui_active():
            self._refresh_board()

        if not self.game_over and self.is_running:
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
        explore = not isinstance(self.agent, MinimaxAgent)
        return int(self.agent.choose_action(state, legal_moves, explore=explore))

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
            return
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
        row, col = divmod(int(action), self.env.board_size)
        was_opening = self.env.move_count < max(4, self.env.win_length)
        state = self.env.get_perspective_board(player)
        result = self.env.step(action)
        next_state = self.env.get_perspective_board(-player) if not result.done else np.zeros_like(state)
        transition = Transition(state=state, action=int(action), reward=float(result.reward), next_state=next_state, done=bool(result.done), board_size=self.env.board_size)

        ai_player = 1 if self.human_side_var.get() == "Second" else -1
        if self.mode_var.get() == "Training":
            store_current = self.training_opponent_var.get() != "Random Bot" or player == 1
        else:
            store_current = player == ai_player

        if store_current and was_opening and not result.info.get("illegal"):
            center = (self.env.board_size - 1) / 2.0
            radius = max(1.5, self.env.board_size * 0.22)
            distance = float(np.sqrt((row - center) ** 2 + (col - center) ** 2))
            self.center_opening_samples += 1
            if distance <= radius:
                self.center_opening_hits += 1

        if learning_enabled or self.mode_var.get() == "Human vs AI":
            self._process_learning(player, transition, result, store_current=store_current)

        self.total_moves += 1
        self._ui_tick_counter += 1
        self.last_move = result.info.get("move") if isinstance(result.info, dict) else None
        self.game_over = bool(result.done)

        no_ui_active = self._is_no_ui_active()

        if result.info.get("illegal"):
            if not no_ui_active:
                self.status_var.set("Illegal move: -50 and turn skipped.")
        elif result.info.get("winner"):
            winner = int(result.info["winner"])
            self.status_var.set("Player 1 wins." if winner == 1 else "Player 2 wins.")
            self.win_line_cells = result.info.get("win_line")
            self.show_win_line = False
        elif result.done:
            self.status_var.set("Draw: board is full.")
            self.win_line_cells = None
            self.show_win_line = False
        elif not no_ui_active:
            move_name = "X" if self.env.current_player == 1 else "O"
            self.status_var.set(f"Move played. Next turn: {move_name}.")

        if no_ui_active:
            if result.done or (self._ui_tick_counter % 20 == 0):
                self._refresh_turn_label()
                self._update_statistics()
        else:
            self._refresh_turn_label()
            self._update_statistics()
        if self.mode_var.get() == "Human vs AI" and not self.game_over and self._is_ai_turn():
            self.after(int(self.move_delay_var.get()), self._auto_ai_step)

        if self.game_over and not self.finish_scheduled:
            self.finish_scheduled = True
            if self.env.winner != 0 and self.win_line_cells:
                self.after(self.win_line_delay_ms, self._show_win_line)
                self.after(self.win_line_delay_ms + max(320, int(self.move_delay_var.get())), self._finish_episode)
            else:
                self.after(max(220, int(self.move_delay_var.get())), self._finish_episode)

    def _show_win_line(self) -> None:
        self.show_win_line = True
        self._refresh_board()

    def _process_learning(self, player: int, transition: Transition, result, store_current: bool) -> None:
        if isinstance(self.agent, MinimaxAgent):
            return
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
        if not self.finish_scheduled:
            return

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
        self._ui_tick_counter = 0
        self._update_statistics()
        self.pending_transitions = {1: None, -1: None}
        self.finish_scheduled = False
        self.game_over = False
        self.win_line_cells = None
        self.show_win_line = False
        self.env.reset()
        self._refresh_board()
        self._refresh_turn_label()
        if self.mode_var.get() == "Training" and self.is_running:
            self._schedule_next_step(240)
        if self.mode_var.get() == "Human vs AI" and self._is_ai_turn():
            self.after(250, self._auto_ai_step)

    def _refresh_turn_label(self) -> None:
        if self.mode_var.get() == "Human vs AI":
            human_player = 1 if self.human_side_var.get() == "First" else -1
            turn = "Human" if int(self.env.current_player) == human_player else "AI"
            color = "X" if self.env.current_player == 1 else "O"
            self.turn_var.set(f"Current turn: {turn} ({color})")
        else:
            color = "X" if self.env.current_player == 1 else "O"
            self.turn_var.set(f"Current turn: {color}")

    def _update_statistics(self) -> None:
        epsilon = getattr(self.agent, "epsilon", 1.0)
        updates = getattr(self.agent, "total_updates", 0)
        self.stats_var.set(f"Episodes: {self.total_episodes} | Moves: {self.total_moves} | Updates: {updates} | Epsilon: {epsilon:.3f}")
        if self.mode_var.get() == "Human vs AI":
            self.score_var.set(f"AI Wins: {self.ai_wins} | AI Losses: {self.ai_losses} | Draws: {self.draws}")
        else:
            self.score_var.set(f"P1 Wins: {self.p1_wins} | P2 Wins: {self.p2_wins} | Draws: {self.draws}")

        self.x_score_var.set(str(self.p1_wins))
        self.o_score_var.set(str(self.p2_wins))
        self.ai_level_var.set(f"{self._estimate_ai_level(float(epsilon), int(updates))}/100")
        center_ratio = 0.0 if self.center_opening_samples == 0 else (self.center_opening_hits / self.center_opening_samples) * 100.0
        self.center_focus_var.set(f"Center focus: {center_ratio:.1f}% (opening AI moves)")

    def _estimate_ai_level(self, epsilon: float, updates: int) -> int:
        if isinstance(self.agent, MinimaxAgent):
            depth_boost = min(100.0, 55.0 + 12.0 * float(self.agent.max_depth))
            return int(round(depth_boost))
        update_score = min(60.0, updates / 50.0)
        confidence_score = max(0.0, 25.0 * (1.0 - min(1.0, epsilon)))
        center_ratio = 0.0 if self.center_opening_samples == 0 else (self.center_opening_hits / self.center_opening_samples) * 100.0
        center_score = min(15.0, center_ratio * 0.15)
        return int(round(min(100.0, update_score + confidence_score + center_score)))

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
        for row in range(size):
            for col in range(size):
                value = int(self.env.board[row, col])
                if value == 0:
                    continue
                cx = margin + col * cell + cell / 2
                cy = margin + row * cell + cell / 2
                if value == 1:
                    self._draw_x(cx, cy, cell)
                else:
                    self._draw_o(cx, cy, cell)

        if self.env.last_move is not None:
            row, col = self.env.last_move
            cx = margin + col * cell + cell / 2
            cy = margin + row * cell + cell / 2
            radius = cell * 0.38
            self.canvas.create_oval(cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6, outline="#FF9F43", width=3)

        if self.show_win_line and self.win_line_cells and len(self.win_line_cells) >= 2:
            first_row, first_col = self.win_line_cells[0]
            last_row, last_col = self.win_line_cells[-1]
            x1 = margin + first_col * cell + cell / 2
            y1 = margin + first_row * cell + cell / 2
            x2 = margin + last_col * cell + cell / 2
            y2 = margin + last_row * cell + cell / 2
            self.canvas.create_line(x1, y1, x2, y2, fill="#F7D354", width=max(4, int(cell * 0.18)), capstyle="round")

        self.canvas.configure(bg="#D7B57D")

    def _draw_x(self, cx: float, cy: float, cell: float) -> None:
        half = cell * 0.32
        self.canvas.create_line(cx - half, cy - half, cx + half, cy + half, fill="#E74C3C", width=max(2, int(cell * 0.12)), capstyle="round")
        self.canvas.create_line(cx - half, cy + half, cx + half, cy - half, fill="#E74C3C", width=max(2, int(cell * 0.12)), capstyle="round")

    def _draw_o(self, cx: float, cy: float, cell: float) -> None:
        radius = cell * 0.34
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#3498DB",
            width=max(2, int(cell * 0.11)),
        )

    def _draw_labels(self, size: int, margin: int, cell: float) -> None:
        for index in range(size):
            label = str(index + 1)
            x = margin + index * cell + cell / 2
            y = margin / 2
            self.canvas.create_text(x, y, text=label, fill="#3A2C12", font=("Segoe UI", 9, "bold"))
            self.canvas.create_text(margin / 2, margin + index * cell + cell / 2, text=label, fill="#3A2C12", font=("Segoe UI", 9, "bold"))

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
            elif model_type == "minimax":
                loaded = MinimaxAgent.load(file_path)
            else:
                raise ValueError("Unsupported model file")

            self.env = GomokuEnv(board_size=board_size)
            self.agent = loaded
            if isinstance(self.agent, QLearningAgent):
                self.algorithm_var.set("Q-Learning")
            elif isinstance(self.agent, DQNAgent):
                self.algorithm_var.set("DQN")
            elif isinstance(self.agent, MinimaxAgent):
                self.algorithm_var.set("Minimax")
            else:
                self.algorithm_var.set("Auto")
            self.board_size_var.set(board_size)
            self.board_size_display_var.set(f"{board_size} x {board_size}")
            self.board_slider.set(board_size)
            self.pending_transitions = {1: None, -1: None}
            self.game_over = False
            self.is_running = False
            self.finish_scheduled = False
            self.win_line_cells = None
            self.show_win_line = False
            self._refresh_start_icon()
            if hasattr(self.agent, "set_board_size"):
                self.agent.set_board_size(board_size)
            self.status_var.set(f"Loaded model from {os.path.basename(file_path)}")
            self._refresh_board()
            self._refresh_turn_label()
            self._update_statistics()
        except Exception as exc:  # pragma: no cover - user feedback path
            messagebox.showerror("Load failed", str(exc))


def run_app() -> None:
    app = GomokuApp()
    app.mainloop()
