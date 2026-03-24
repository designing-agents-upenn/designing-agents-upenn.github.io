import os
from typing import List, Optional

from mistralai import Mistral  # pyright: ignore[reportMissingImports]

API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral-small-latest"

def query_mistral(prompt: str, temperature: float = 0.0) -> str:
    if not API_KEY:
        raise ValueError("Missing MISTRAL_API_KEY.")
    client = Mistral(api_key=API_KEY)
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response.choices[0].message.content).strip()


def has_won(board: List[str], symbol: str) -> bool:
    lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    return any(board[a] == board[b] == board[c] == symbol for a, b, c in lines)


def show_game(board: List[str]) -> None:
    print("\nCurrent board:")
    for r in range(3):
        i = r * 3
        print(" " + " | ".join(board[i : i + 3]))
        if r < 2:
            print("---+---+---")
    print()


def human_turn(board: List[str], human_symbol: str = "X") -> None:
    while True:
        value = input("Choose a square (1-9): ").strip()
        if not value.isdigit():
            print("Please enter a number from 1 to 9.")
            continue
        move = int(value) - 1
        if move not in range(9):
            print("Out of range. Use 1 to 9.")
            continue
        if board[move] != " ":
            print("That square is taken. Try again.")
            continue
        board[move] = human_symbol
        return


def agent_turn(board: List[str], human_symbol: str = "X", agent_symbol: str = "O") -> Optional[str]:
    # 1) Observe board: did human already win?
    if has_won(board, human_symbol):
        return "Congrats! Good game"

    def board_text() -> str:
        cells = [str(i + 1) if board[i] == " " else board[i] for i in range(9)]
        return (
            f"{cells[0]} | {cells[1]} | {cells[2]}\n"
            f"{cells[3]} | {cells[4]} | {cells[5]}\n"
            f"{cells[6]} | {cells[7]} | {cells[8]}"
        )

    def available_moves() -> List[int]:
        return [i for i, v in enumerate(board) if v == " "]

    def parse_move_list(text: str) -> List[int]:
        moves: List[int] = []
        token = ""
        for ch in text:
            if ch.isdigit():
                token += ch
            elif token:
                moves.append(int(token))
                token = ""
        if token:
            moves.append(int(token))
        return moves

    def first_int(text: str) -> Optional[int]:
        token = ""
        for ch in text:
            if ch.isdigit():
                token += ch
            elif token:
                break
        return int(token) if token else None

    # 2) Determine possible moves.
    q2 = (
        f"Board:\n{board_text()}\n"
        f"Human is {human_symbol}. Agent is {agent_symbol}.\n"
        "List all empty squares as 0-based indices only, comma-separated. "
        "Example: 0,3,8"
    )
    moves = parse_move_list(query_mistral(q2))
    valid = available_moves()
    moves = [m for m in moves if m in valid]
    if not moves:
        moves = valid
    if not moves:
        return "Draw."

    # 3) Decide the best move.
    q3 = (
        f"Board:\n{board_text()}\n"
        f"Possible moves: {moves}\n"
        f"You are {agent_symbol}. Pick the best move. "
        "Reply with one 0-based index only."
    )
    chosen = first_int(query_mistral(q3, temperature=0.2))
    if chosen in moves:
        move = chosen
    else:
        move = 4 if 4 in moves else moves[0]
    board[move] = agent_symbol
    print(f"Agent plays square {move + 1}.")

    # 4) Observe board: did agent win?
    if has_won(board, agent_symbol):
        return "I beat you!"
    return None


def play_game() -> None:
    board = [" "] * 9
    print("Welcome to Tic-Tac-Toe! You are X, the agent is O.")
    show_game(board)

    while True:
        human_turn(board, human_symbol="X")
        show_game(board)

        try:
            result = agent_turn(board, human_symbol="X", agent_symbol="O")
        except Exception as e:
            print(f"Agent turn failed: {e}")
            break
        if result:
            print(result)
            break

        show_game(board)
        if not any(cell == " " for cell in board):
            print("Draw.")
            break


if __name__ == "__main__":
    play_game()
