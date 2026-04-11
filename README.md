# Gomoku Minimax Trainer

Ứng dụng Caro/Gomoku dùng Minimax + Alpha-Beta với giao diện `customtkinter`.

## Tính năng chính

- Bàn cờ từ 3x3 đến 15x15.
- Win length tự động: 3 cho bàn 3x3, 4 cho bàn 4x4, và 5 cho các bàn lớn hơn.
- Hai chế độ: Training và Human vs AI.
- Training opponent: Self-Play hoặc Random Bot.
- AI Minimax + Alpha-Beta cho chế độ chơi ổn định.
- Lưu / tải model bằng file pickle.

## Cài đặt

Tạo môi trường ảo và cài dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy chương trình

Chạy trực tiếp GUI (mặc định):

```powershell
python main.py
```

## Ghi chú kỹ thuật

- Engine đã khóa sang `MinimaxAgent` + Alpha-Beta.
- Heuristic evaluation mới dùng pattern scoring (do dai day + so dau mo + center control).
- Khong dung tactical override theo kieu cu (block/open-three rule hardcode) trong duong Minimax.
- UI đã khóa chế độ thuật toán để tránh lệch backend.
- Khi người chơi click ô đã có quân, nước đi bị chặn ngay ở UI.
- Khi đổi kích thước bàn cờ, model sẽ được khởi tạo lại vì kích thước state/action thay đổi.

## Mẹo sử dụng

1. Chọn `Training` để AI tự đấu.
2. Chỉnh `Board Size` xuống 3 hoặc 5 trước để curriculum học nhanh hơn.
3. Dùng `Speed Slider` để xem quá trình AI suy nghĩ.
4. Trong `Human vs AI`, quân người được hiển thị trước, sau đó AI mới bắt đầu tính nước.
