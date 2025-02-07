from helperfuncs import *
import numpy as np

try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
    print("5 piece Syzygy endgame tablebase found.")
except:
    print("Could not find tablebase")
    TABLEBASE = None

TIMES_UP = -999

class Node:
    pass

class Node:

    def __init__(self, board: chess.Board, move: chess.Move | None, net: nn.Module, parent: Node | None, depth=0):
        self.board = board
        self.move = move
        self.value = 0.5
        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []

    def ucb(self, c=1.4):
        try:
            if self.board.turn:
                return (self.value / (self.visits + 1)) + c * (np.log(self.parent.visits) / (self.visits + 1))
            elif not self.board.turn:
                return (self.value / (self.visits + 1)) - c * (np.log(self.parent.visits) / (self.visits + 1))
        except:
            return self.value / (self.visits + 1)
    
    def evaluate_position(self):
        if TABLEBASE and lt5(self.board):
            result = TABLEBASE.probe_wdl(self.board)
            if result == 2:
                return self.board.turn
            elif result == -2:
                return not self.board.turn
            elif result in [-1, 0, 1]:
                return 0.5
        outcome = self.board.result(claim_draw=True)
        if outcome != "*":
            if outcome == "1-0":
                return chess.WHITE + max((10 - self.depth) / 10, 0)
            elif outcome == "0-1":
                return chess.BLACK - max((10 - self.depth) / 10, 0)
            elif outcome == "1/2-1/2":
                return 0.5
        return None
    
    def generate_children(self):
        all_positions = []
        all_feats = []

        evaled = []
        not_evaled = []

        for move in self.board.legal_moves:
            newboard = self.board.copy()
            newboard.push(move)
            newnode = Node(newboard, move, self.net, self, self.depth + 1)
            score = newnode.evaluate_position()
            if score is not None:
                newnode.value = score
                evaled.append(newnode)
            else:
                all_positions.append(fast_board_to_boardmap(newboard))
                all_feats.append(fast_board_to_feature(newboard))
                not_evaled.append(newnode)

        pos = torch.tensor(all_positions, device=device, dtype=torch.float).reshape(len(not_evaled), 1, 8, 8)
        feat = torch.tensor(all_feats, device=device, dtype=torch.float).reshape(len(not_evaled), 12)
        result = self.net.forward(pos, feat)
        for i in range(len(not_evaled)):
            not_evaled[i].value = float(result[i])
            evaled.append(not_evaled[i])
        
        self.children = evaled


    def mcts(self, start_time, time_for_this_move, c=1.4):
        while time.time() - start_time < time_for_this_move:

            # 1. Traverse tree
            target_node = self
            while target_node.children != []:
                target_node.visits += 1
                if target_node.board.turn:
                    target_node.children = sorted(target_node.children, key=lambda child: child.ucb(c), reverse=True)
                elif not target_node.board.turn:
                    target_node.children = sorted(target_node.children, key=lambda child: child.ucb(c))
                target_node = target_node.children[0]
            
            # 2. Expansion
            target_node.generate_children()

            # 3. Backpropagation
            if len(target_node.children) == 0:
                best_value = target_node.value
            else:
                if target_node.board.turn:
                    best_value = max(target_node.children, key=lambda child: child.value).value
                elif target_node.board.turn:
                    best_value = min(target_node.children, key=lambda child: child.value).value

            while target_node.parent is not None:
                target_node = target_node.parent
                target_node.value = best_value

        # 4. Select move
        if target_node.board.turn:
            selected_child = max(self.children, key=lambda child: (0.8) * (child.visits / self.visits) + 0.2 * child.value)
        elif not target_node.board.turn:
            selected_child = min(self.children, key=lambda child: -(0.8) * (child.visits / self.visits) + 0.2 * child.value)

        return selected_child



