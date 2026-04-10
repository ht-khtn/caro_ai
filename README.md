# Gomoku RL Trainer

Ứng dụng huấn luyện Caro/Gomoku bằng reinforcement learning với giao diện `customtkinter`.

## Tính năng chính

- Bàn cờ từ 3x3 đến 15x15.
- Win length tự động: 3 cho bàn 3x3, 4 cho bàn 4x4, và 5 cho các bàn lớn hơn.
- Hai chế độ: Training và Human vs AI.
- Training opponent: Self-Play hoặc Random Bot.
- AI tự học liên tục trong cả chế độ training và khi chơi với người.
- Lưu / tải model bằng file pickle.
- Reward shaping theo yêu cầu:
  - Thắng: +100
  - Thua: -100
  - Chặn đối thủ 4-in-a-row: +40
  - Chặn đối thủ 3-in-a-row: +20
  - Tạo 4-in-a-row: +25
  - Tạo 3-in-a-row: +15
  - Tạo 2-in-a-row: +2
  - Nước đi hợp lệ nhưng sai vị trí / không hợp lệ: -50 và bỏ lượt

## Cài đặt

Tạo môi trường ảo và cài dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy chương trình

```powershell
python main.py
```

## Ghi chú kỹ thuật

- `QLearningAgent` dùng cho bàn nhỏ.
- `DQNAgent` dùng mạng MLP thuần NumPy cho bàn lớn.
- Augmentation đối xứng 8 hướng được áp dụng để tăng dữ liệu huấn luyện.
- Khi đổi kích thước bàn cờ, model sẽ được khởi tạo lại vì kích thước state/action thay đổi.

## Mẹo sử dụng

1. Chọn `Training` để AI tự đấu và học.
2. Chỉnh `Board Size` xuống 3 hoặc 5 trước để curriculum học nhanh hơn.
3. Dùng `Speed Slider` để xem quá trình AI suy nghĩ.
4. Trong `Human vs AI`, AI vẫn tiếp tục học từ các ván đấu của bạn.
