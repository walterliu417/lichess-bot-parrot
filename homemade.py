"""
This file is designed to interact with the Colab notebook parrot_implementation.ipynb.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
import time

from helperfuncs import *
from tree import *
from nn_creator import *

# Logging is useless in Google Colab, replaced with print.


try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
except:
    print("Could not find 5-piece tablebase.")
    TABLEBASE = None
    

class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""


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
        
        model_name = "new_parrot"
        self.model = SimpleModel(model_name)
        state = torch.load(f"/content/drive/MyDrive/parrot/best_{model_name}.pickle", weights_only=True, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        print(self.model)
        print("Current best model loaded successfully!")

    def search(self, board: chess.Board, time_limit: chess.engine.Limit, *args) -> PlayResult:
        # Hybrid MCTS + alpha-beta search.
        if time_limit.time:
            self.time_for_this_move = time_limit.time
            self.time_remaining = time_limit.time * 60
        else:
            if board.turn == chess.WHITE:
                self.time_remaining = time_limit.white_clock
            elif board.turn == chess.BLACK:
                self.time_remaining = time_limit.black_clock

            # Simple time management
            if board.fullmove_number < 5:
                # Opening - save time
                self.time_for_this_move = self.time_remaining / 60
            else:
                self.time_for_this_move = (self.time_remaining / 20) * 0.925 # Safety factor for network issues, etc
        print(f"Time remaining: {self.time_remaining} seconds.")
        print(f"Starting search for {self.time_for_this_move} seconds.")
        search_start = time.time()
        
        root_node = Node(board, None, self.model, None)
        selected_child = root_node.mcts(search_start, self.time_for_this_move)
        print(f"Visited {root_node.visits} times for a nps of {root_node.visits / self.time_for_this_move}")
        print(f"Evaluation: {selected_child.value}")
        
        try:
            return PlayResult(selected_child.move, None)
        except:
            # Fail-safe: return random move :(
            return PlayResult(random.choice(list(board.legal_moves)), None)
