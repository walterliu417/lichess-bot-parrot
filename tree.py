class Node:

    def __init__(self, board):
        self.board = board
        self.value = 0.5
        self.transposition_children = []
        self.capture_children = []
        self.check_children = []
        self.all_other_children = []
        

    def evaluate(self, board: chess.Board):
        pos = torch.FloatTensor(board_to_boardmap(board)).unsqueeze(0)
        feat = torch.FloatTensor(fen_to_feature_wboard_list(board.fen())).unsqueeze(0)
        val = float(model.forward(pos, feat))
        return val