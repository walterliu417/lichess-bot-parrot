from helperfuncs import *

class Node:

    def __init__(self, board):
        self.board = board
        self.value = 0.5
        self.transposition_children = []
        self.capture_children = []
        self.check_children = []
        self.all_other_children = []
        

    def evaluate(self, net: torch.nn.Module):
        pos = torch.FloatTensor(board_to_boardmap(self.board), device=device).unsqueeze(0)
        feat = torch.FloatTensor(fen_to_feature_wboard_list(self.board.fen()), device=device).unsqueeze(0)
        val = float(net.forward(pos, feat))
        return val