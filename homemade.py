"""
This file is designed to interact with the Colab notebook parrot_implementation.ipynb.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import time

from tree import *

# Logging is useless in Google Colab, replaced with print.



try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
except:
    print("Could not find 5-piece tablebase.")
    TABLEBASE = None
    

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)

class Parrot(ExampleEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_remaining = 0
        self.time_control = 0

    def search(self, board: chess.Board, time_limit: chess.engine.Limit, *args) -> PlayResult:
        # Hybrid MCTS + alpha-beta search.
        if time_limit.time:
            self.time_for_this_move = time_limit.time
            self.time_control = time_limit.time
        else:
            if board.turn == chess.WHITE:
                self.time_remaining = time_limit.white_clock
            elif board.turn == chess.BLACK:
                self.time_remaining = time_limit.black_clock

            # Simple time management
            if board.fullmove_number < 5:
                # Opening - save time
                self.time_for_this_move = self.time_control / 100
            else:
                if board.fullmove_number < 40:
                    expected_moves_left = 60 - board.fullmove_number
                elif board.fullmove_number < 80:
                    expected_moves_left = 100 - board.fullmove_number
                else:
                    expected_moves_left = 20
                self.time_for_this_move = (self.time_remaining / expected_moves_left) * 0.925 # Safety factor for network issues, etc

        search_start = time.time()
        
        while time.time() - search_start < self.time_for_this_move:
            # Do something to evaluate the position...
            pass
        
        try:
            return PlayResult(best_move, None)
        except:
            # Fail-safe: return random move :(
            return PlayResult(random.choice(list(board.legal_moves)), None)
