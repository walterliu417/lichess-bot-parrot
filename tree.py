from helperfuncs import *

try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
    print("5 piece Syzygy endgame tablebase found.")
except:
    print("Could not find tablebase")
    TABLEBASE = None

class Node:

    def __init__(self, board: chess.Board, net: nn.Module, parent_depth, table: dict):
        self.board = board
        self.parent_depth = parent_depth    
        self.value = 0.5
        self.transposition_children = []
        self.capture_children = []
        self.check_children = []
        self.all_other_children = []

        self.net = net
        self.table = table

    def evaluate_nn(self):
        pos = torch.tensor(board_to_boardmap(self.board), device=device, dtype=torch.float).unsqueeze(0)
        feat = torch.tensor(fen_to_feature_wboard_list(self.board.fen()), device=device, dtype=torch.float).unsqueeze(0)
        val = float(self.net.forward(pos, feat))
        return val
    
    def evaluate_position(self, depth):
        winner = False
        outcome = self.board.result(claim_draw=True)
        if outcome != "*":
            if outcome == "1-0":
                winner = chess.WHITE + max((self.parent_depth - depth) / 10, 0)
            elif outcome == "0-1":
                winner = chess.BLACK - max((self.parent_depth - depth) / 10, 0)
            elif outcome == "1/2-1/2":
                winner = 0.5
        elif len(self.board.piece_map()) <= 5 and TABLEBASE:
            result = TABLEBASE.probe_wdl(self.board)
            if result == 2:
                winner = self.board.turn
            elif result == -2:
                winner = not self.board.turn
            elif result in [-1, 0, 1]:
                winner = 0.5
        return winner

    
    def alpha_beta(self, alpha, beta, depth, start_time, time_for_this_move):

        if time.time() - start_time > time_for_this_move:
            return None # Times up
        
        winner = self.evaluate_position(depth)
        if winner is not False:
            return winner
        
        if depth == 0:
            # Leaf node.
            return self.evaluate_nn()
        
        if (not self.transposition_children) and (not self.capture_children) and (not self.check_children) and (not self.all_other_children):
            moves = list(self.board.legal_moves)
            for move in moves:
                if self.board.is_capture(move):
                    self.capture_children.append(move)
                elif self.board.is_check(move):
                    self.check_children.append(move)
                else:
                    self.all_other_children.append(move)

        if self.board.turn:
            # White player - maximising.
            best_value = -10000
            for node in self.transposition_children:
                value = node.alpha_beta(alpha, beta, depth - 1, start_time, time_for_this_move)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    # Beta cutoff
                    break

            for move in self.capture_children:
                new_board = self.board.copy(stack=False)
                new_board.push(move)
                new_node = Node(new_board, self.net, self.parent_depth, self.table)
                value = node.

            