# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        maxi = float('-inf')
        res = None
        for action in gameState.getLegalActions(0):
            value = self.getMin(gameState.generateSuccessor(0, action))
            if value is not None and value > maxi:
                maxi = value
                res = action
        return res

    def getMax(self, gameState, depth=0, agentIndex=0):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        maxi = float('-inf')
        for action in actions:
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and value > maxi:
                maxi = value
        return maxi

    def getMin(self, gameState, depth=0, agentIndex=1):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        mini = float('inf')
        for action in actions:
            if agentIndex == gameState.getNumAgents() - 1:
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value is not None and value < mini:
                mini = value
        return mini


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.getMax(gameState)[1]

    def getMax(self, gameState, depth=0, agentIndex=0, alpha=float('-inf'), beta=float('inf')):
        # 如果到达叶子节点，或者无法继续展开，则返回当前状态的评价值
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState), None
        # 否则，就继续往下遍历吃豆人可能的下一步
        maxi = float('-inf')
        res = None
        for action in actions:
            # 考虑只有一个吃豆人的情况，直接求其MIN分支的评价值，agentIndex从1开始遍历所有鬼怪
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)[0]
            if value is not None:
                if value > maxi:
                    maxi = value
                    res = action
                # 按照α-β剪枝算法，如果v>β，则直接返回v
                if value > beta:
                    return value, action
                # 按照α-β剪枝算法，这里还需要更新α的值
                if value > alpha:
                    alpha = value
        return maxi, res

    def getMin(self, gameState, depth=0, agentIndex=0, alpha=float('-inf'), beta=float('inf')):
        # 如果到达叶子节点，或者无法继续展开，则返回当前状态的评价值
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState), None
        # 否则，就继续往下遍历当前鬼怪可能的下一步
        mini = float('inf')
        res = None
        for action in legalActions:
            # 如果当前是最后一个鬼怪，那么下一轮就该调用MAX函数
            if agentIndex >= gameState.getNumAgents() - 1:
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta)[0]
            else:
                # 如果不是最后一个鬼怪，则继续遍历下一个鬼怪，即agentIndex+1
                value = \
                    self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)[0]
            if value is not None:
                if value < mini:
                    mini = value
                    res = action
                if value < alpha:
                    # 按照α-β剪枝算法，如果v<α，则直接返回v
                    if value is not None and value < alpha:
                        return value, action
                if value < beta:
                    beta = value
        return mini, res


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.getMax(gameState)

    def getMax(self, gameState, depth=0, agentIndex=0):
        # 获得吃豆人所有下一步行动
        actions = gameState.getLegalActions(agentIndex)
        # 如果到达根节点或者没有可行的行动，则返回评价函数值
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        # 否则初始化，并对合法的下一步进行遍历
        maxi = float('-inf')
        bestAction = None
        for action in actions:
            # 从第一个鬼怪开始，进行Expectimax操作
            value = self.getExpectation(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and value > maxi:
                maxi = value
                bestAction = action
        if depth is 0 and agentIndex is 0:
            return bestAction
        else:
            return maxi

    def getExpectation(self, gameState, depth=0, agentIndex=0):
        actions = gameState.getLegalActions(agentIndex)
        # 如果到达根节点，或者没有下一步了，则返回评价函数值
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
            # 初始化效用值总计
        exp = 0
        actionNum = len(actions)
        # 轮询当前鬼怪所有可行的下一步
        for action in actions:
            # 同样，如果是最后一个鬼怪，那么接下来要去算吃豆人的MAX值
            if agentIndex >= gameState.getNumAgents() - 1:
                exp += self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            # 否则，挨个遍历各个鬼怪，计算Expectation值，并计入效用总计
            else:
                exp += self.getExpectation(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
        # 最后需要把所有可能的下一步的效用值求平均，并返回
        return exp / float(actionNum)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    # 以当前游戏状态为准,评价函数由距离最近的食物颗粒的距离给出，如果没有颗粒，则为0
    eval = currentGameState.getScore()
    foodDist = float("inf")
    for food in newFood:
        foodDist = min(foodDist, util.manhattanDistance(food, newPos))
    eval += 1.0 / foodDist

    return eval


# Abbreviation
better = betterEvaluationFunction
