from helperfuncs import *
import numpy as np

try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
    print("5 piece Syzygy endgame tablebase found.")
except:
    print("Could not find tablebase")
    TABLEBASE = None

TIMES_UP = -999
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class Node:
    pass

class Node:

    def __init__(self, board: chess.Board, move: chess.Move | None, net: nn.Module, parent: Node | None, depth=0):
        self.board = board
        self.move = move
        self.value = None
        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []
        self.flag = None

    def ucb(self, c=1.4):
        try:
            if self.board.turn:
                return (self.value / (self.visits + 1)) + c * (np.log(self.parent.visits) / (self.visits + 1))
            elif not self.board.turn:
                return (self.value / (self.visits + 1)) - c * (np.log(self.parent.visits) / (self.visits + 1))
        except:
            return self.value / (self.visits + 1)
        
    def evaluate_nn(self):
        pos = torch.tensor(fast_board_to_boardmap(self.board), device=device, dtype=torch.float).reshape(1, 1, 8, 8)
        feat = torch.tensor(fast_board_to_feature(self.board), device=device, dtype=torch.float).reshape(1, 12)
        return self.net.forward(pos, feat)
    
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
            if self.table: 
                newnode.flag = EXACT
                self.table[newboard.fen()] = newnode

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
    
    def negamax(self, depth, alpha, beta, color, start_time, time_for_this_move):
        if time.time() - start_time > time_for_this_move:
            return TIMES_UP, None
        alpha_original = 0
        
        if (self.value is not None) and depth == 0:
            if self.flag == EXACT:
                return self.value, self.move
            elif self.flag == LOWERBOUND:
                alpha = max(alpha, self.value)
            elif self.flag == UPPERBOUND:
                beta = min(beta, self.value)
            
            if alpha >= beta:
                return self.value, self.move
        

        value = self.evaluate_position()
        if value is not None:
            return value * color, self.move
        if depth == 0:
            return self.evaluate_nn() * color, self.move
        
        if self.children != []:
            evaled, not_evaled = [], []
            for child in self.children:
                if child.value is not None:
                    evaled.append(child)
                else:
                    not_evaled.append(child)
            self.children = evaled + not_evaled

        elif self.children == []:
            captures = []
            checks = []
            others = []
            for move in self.board.legal_moves:
                newboard = self.board.copy()
                newboard.push(move)
                newnode = Node(newboard, move, self.net, self, self.depth + 1)
                if self.board.is_capture(move):
                    captures.append(newnode)
                elif self.board.gives_check(move):
                    checks.append(newnode)
                else:
                    others.append(newnode)
            self.children = captures + checks + others

        value = -10000
        best_child = None
        for child in self.children:
            newvalue, newmove = child.negamax(depth - 1, -beta, -alpha, -color, start_time, time_for_this_move)
            if newvalue == TIMES_UP:
                return TIMES_UP, None
            if -newvalue > value:
                value = -newvalue
                best_child = child
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        self.value = value
        if value <= alpha_original:
            self.flag = UPPERBOUND
        elif value >= beta:
            self.flag = LOWERBOUND
        else:
            self.flag = EXACT
        
        return value, best_child.move





