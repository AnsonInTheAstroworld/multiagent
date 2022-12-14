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
            # ????????????????????????????????????
            value = self.getMin(gameState.generateSuccessor(0, action))
            if value is not None and value > maxi:
                # ????????????????????????
                maxi = value
                res = action
        return res

    def getMax(self, gameState, depth=0, agentIndex=0):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)

        maxi = float('-inf')
        for action in actions:
            # ????????????????????????????????????
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and value > maxi:
                # ???????????????
                maxi = value
        return maxi

    def getMin(self, gameState, depth=0, agentIndex=1):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)

        mini = float('inf')
        for action in actions:
            if agentIndex == gameState.getNumAgents() - 1:
                #???????????????????????????????????????????????????getMax
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if value is not None and value < mini:
                # ???????????????
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
        maxi = float('-inf')
        res = None
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            value = self.getMin(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value is not None:
                if value > maxi:
                    maxi = value
                    res = action
                # ???value > beta?????????beta
                if value > beta:
                    return res
                if value > alpha:
                    # ???value > alpha?????????alpha
                    alpha = value
        return res

    def getMax(self, gameState, depth, agentIndex, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)

        maxi = float('-inf')
        for action in actions:
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)
            if value is not None:
                if value > maxi:
                    maxi = value
                if value > beta:
                    # ???value > beta?????????
                    return value
                if value > alpha:
                    # ???value > alpha?????????alpha
                    alpha = value
        return maxi

    def getMin(self, gameState, depth, agentIndex, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)

        mini = float('inf')
        for action in legalActions:
            if agentIndex >= gameState.getNumAgents() - 1:
                # ???????????????????????????????????????????????????getMax
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta)
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)
            if value is not None:
                if value < mini:
                    mini = value
                if value < alpha:
                    # ???value<alpha?????????
                    return value
                if value < beta:
                    # ???value<beta????????????beta
                    beta = value
        return mini


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
        maxi = float('-inf')
        res = None
        for action in gameState.getLegalActions(0):
            value = self.getExpectedUtility(gameState.generateSuccessor(0, action), 0)
            # ????????????????????????
            if value is not None and value > maxi:
                maxi = value
                res = action
        return res

    def getMax(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)

        maxi = float('-inf')
        for action in actions:
            # ?????????????????????????????????
            value = self.getExpectedUtility(gameState.generateSuccessor(agentIndex, action), depth)
            # ???????????????
            if value is not None and value > maxi:
                maxi = value
        return maxi

    def getExpectedUtility(self, gameState, depth, agentIndex=1):
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)

        exp = 0
        for action in actions:
            if agentIndex >= gameState.getNumAgents() - 1:
                # ??????????????????????????????????????????????????????????????????getMax
                exp += self.getMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
            else:
                # ????????????????????????????????????
                exp += self.getExpectedUtility(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

        # ???????????????????????????
        return exp / float(len(actions))


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = [food for food in currentGameState.getFood().asList() if food]
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghost.scaredTimer for ghost in newGhostStates]

    # ????????????????????????
    foodDist = float('inf')
    for food in newFood:
        foodDist = min(manhattanDistance(food, newPos), foodDist)
    foodDistScore = 50 / (foodDist+1)

    # ??????????????????
    ghostDist = float('inf')
    for ghost in newGhostStates:
        ghostDist=min(ghostDist,manhattanDistance(newPos, ghost.getPosition()))

    # ????????????????????????
    scaredTime = min(newScaredTimes)
    if scaredTime == 0:
        # ???????????????????????????????????????
        ghostDistScore = -100 / (ghostDist+1)
    else:
        # ??????????????????????????????????????????
        ghostDistScore = 50 / (ghostDist+1) * 0.5 + scaredTime * 80

    # ?????????????????????????????????????????????????????????????????????
    return (currentGameState.getScore() * 100) + foodDistScore + ghostDistScore


# Abbreviation
better = betterEvaluationFunction
