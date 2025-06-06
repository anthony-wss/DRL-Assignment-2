import sys
import numpy as np
import random
import mcts
from time import sleep


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.first_move = True

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.first_move = True
        print("= ", flush=True)
    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.first_move = True
        print("= ", flush=True)
    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """
        Places stones and checks the game status.
        `move` is a string representing one or two moves, e.g. "A1,B2"
        """
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        print("We have to move now", color, file=sys.stderr)
        print(self.board, file=sys.stderr, flush=True)

        def print_board():
            for r in range(self.size):
                print(" ".join(str(self.board[r, c]) for c in range(self.size)), file=sys.stderr)
        # print_board()
        if self.game_over:
            print("? Game over")
            return
        
        # print("Generating move for ", color, file=sys.stderr)
        # next_move, _ = mcts.mct_search(self.board, color)

        if self.board.sum() == 0:
            move = random.sample([(r, c) for r in range(8, 10) for c in range(8, 10) if self.board[r, c] == 0], 1)
            move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in move)
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
        else:
            sleep(1)

            # Random two moves
            # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
            # next_move = random.sample(empty_positions, 2)

            # MCTS
            # print("mct search start", file=sys.stderr, flush=True)
            next_move, _ = mcts.mct_search(self.board.copy(), 1 if color == 'B' else 2, debug=True)
            # print("mct search end", file=sys.stderr, flush=True)

            move_str = ",".join([f"{self.index_to_label(c)}{r+1}" for r, c in next_move])
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)

            self.play_move(color, move_str)


        # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        # selected = random.sample(empty_positions, 1)
        # move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
        # self.play_move(color, move_str)

        # print(f"{move_str}\n\n", end='', flush=True)
        # print(move_str, file=sys.stderr)
        # return
    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(flush=True)
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
                print(flush=True)
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
