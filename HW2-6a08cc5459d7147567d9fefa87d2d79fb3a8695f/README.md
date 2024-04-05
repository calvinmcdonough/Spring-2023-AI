## HW2 Due February 10th, 2023
### Part 0. (2 point) Complete this before changing any files.

1. Download a zip of the HW2 repository. (https://github.com/andrewboes/HW2)
2. Upload it to a folder called HW2 in your Artificial-Inteligence-Spring-2023 repository (don't branch first).
3. Create a branch off of your Artificial-Inteligence-Spring-2023 repository on github called “HW2”, do all of your work for this homework in the HW2 branch.

### Part 1. (10 points)

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', complete the solution class in parens.py by using a stack to determine if the input string is valid. The input string is valid if it satisfies all three of the following:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

Example: 
1. Input: s = "()", Output: true
2. Input: s = "([]{})", Output: true
3. Input: s = "(]{)", Output: false
4. Input: s = "[]())", Output: false

### Part 2. (60 points)

Your goal in this part will be to build an AI that can play Minesweeper.

Minesweeper is a puzzle game that consists of a grid of cells, where some of the cells contain hidden “mines.” Clicking on a cell that contains a mine detonates the mine, and causes the user to lose the game. Clicking on a “safe” cell (i.e., a cell that does not contain a mine) reveals a number that indicates how many neighboring cells – where a neighbor is a cell that is one square to the left, right, up, down, or diagonal from the given cell – contain a mine.

In this 3x3 Minesweeper game, for example, the three 1 values indicate that each of those cells has one neighboring cell that is a mine. The four 0 values indicate that each of those cells has no neighboring mine. The "_" value denotes a cell that hasn't been selected.

_&emsp;0&emsp;0  

0&emsp;1&emsp;1

0&emsp;1&emsp;_

Given this information, a logical player could conclude that there must be a mine in the lower-right cell and that there is no mine in the upper-left cell, for only in that case would the numerical labels on each of the other cells be accurate.

#### Your part

There are two main files in this project: runner.py and minesweeper.py. minesweeper.py contains all of the logic the game itself and for the AI to play the game. runner.py has been implemented for you, and contains all of the code to run the graphical interface for the game. Once you’ve completed all the required functions in minesweeper.py, you should be able to run python runner.py to play Minesweeper (or let your AI play for you)!

Let’s open up minesweeper.py to understand what’s provided. There are three classes defined in this file, Minesweeper, which handles the gameplay; Sentence, which represents a logical sentence that contains both a set of cells and a count; and MinesweeperAI, which handles inferring which moves to make based on knowledge.

The Minesweeper class has been entirely implemented for you. Notice that each cell is a pair (i, j) where i is the row number (ranging from 0 to height - 1) and j is the column number (ranging from 0 to width - 1).

The Sentence class will be used to represent logical sentences of the form described in the Background. Each sentence has a set of cells within it and a count of how many of those cells are mines. The class also contains functions known_mines and known_safes for determining if any of the cells in the sentence are known to be mines or known to be safe. It also contains functions mark_mine and mark_safe to update a sentence in response to new information about a cell.

Finally, the MinesweeperAI class will implement an AI that can play Minesweeper. The AI class keeps track of a number of values. self.moves_made contains a set of all cells already clicked on, so the AI knows not to pick those again. self.mines contains a set of all cells known to be mines. self.safes contains a set of all cells known to be safe. And self.knowledge contains a list of all of the Sentences that the AI knows to be true.

The mark_mine function adds a cell to self.mines, so the AI knows that it is a mine. It also loops over all sentences in the AI’s knowledge and informs each sentence that the cell is a mine, so that the sentence can update itself accordingly if it contains information about that mine. The mark_safe function does the same thing, but for safe cells instead.

The remaining functions, add_knowledge, make_safe_move, and make_random_move, are left up to you!

#### Specification

Complete the implementations of the Sentence class and the MinesweeperAI class in minesweeper.py.

In the Sentence class, complete the implementations of known_mines, known_safes, mark_mine, and mark_safe.

* The known_mines function should return a set of all of the cells in self.cells that are known to be mines.
* The known_safes function should return a set of all the cells in self.cells that are known to be safe.
* The mark_mine function should first check to see if cell is one of the cells included in the sentence.
  * If cell is in the sentence, the function should update the sentence so that cell is no longer in the sentence, but still represents a logically correct sentence given that cell is known to be a mine.
  * If cell is not in the sentence, then no action is necessary.
* The mark_safe function should first check to see if cell is one of the cells included in the sentence.
  *  If cell is in the sentence, the function should update the sentence so that cell is no longer in the sentence, but still represents a logically correct sentence given that cell is known to be safe.
  * If cell is not in the sentence, then no action is necessary.
 
In the MinesweeperAI class, complete the implementations of add_knowledge, make_safe_move, and make_random_move.

* add_knowledge should accept a cell (represented as a tuple (i, j)) and its corresponding count, and update self.mines, self.safes, self.moves_made, and self.knowledge with any new information that the AI can infer, given that cell is known to be a safe cell with count mines neighboring it.
  * The function should mark the cell as one of the moves made in the game.
  * The function should mark the cell as a safe cell, updating any sentences that contain the cell as well.
  * The function should add a new sentence to the AI’s knowledge base, based on the value of cell and count, to indicate that count of the cell’s neighbors are mines. Be sure to only include cells whose state is still undetermined in the sentence.
  * If, based on any of the sentences in self.knowledge, new cells can be marked as safe or as mines, then the function should do so.
  * If, based on any of the sentences in self.knowledge, new sentences can be inferred (using the subset method described in the Background), then those sentences should be added to the knowledge base as well.
  * Note that any time that you make any change to your AI’s knowledge, it may be possible to draw new inferences that weren’t possible before. Be sure that those new inferences are added to the knowledge base if it is possible to do so.
* make_safe_move should return a move (i, j) that is known to be safe.
  * The move returned must be known to be safe, and not a move already made.
  * If no safe move can be guaranteed, the function should return None.
  * The function should not modify self.moves_made, self.mines, self.safes, or self.knowledge.
* make_random_move should return a random move (i, j).
  * This function will be called if a safe move is not possible: if the AI doesn’t know where to move, it will choose to move randomly instead.
  * The move must not be a move that has already been made.
  * The move must not be a move that is known to be a mine.
  * If no such moves are possible, the function should return None.

#### Hints

* When implementing known_mines and known_safes in the Sentence class, consider: under what circumstances do you know for sure that a sentence’s cells are safe? Under what circumstances do you know for sure that a sentence’s cells are mines?
* add_knowledge does quite a lot of work, and will likely be the longest function you write for this project by far. It will likely be helpful to implement this function’s behavior one step at a time.
* You’re welcome to add new methods to any of the classes if you would like, but you should not modify any of the existing functions’ definitions or arguments.
* When you run your AI (as by clicking “AI Move”), note that it will not always win! There will be some cases where the AI must guess, because it lacks sufficient information to make a safe move. This is to be expected. runner.py will print whether the AI is making a move it believes to be safe or whether it is making a random move.


### Part 3. (2 point) Do not complete until you are mostly finsihed with the assignment.

Add a text file to the HW1 folder called “feedback.txt” and answer the following:

1. Approximately how many hours did you spend on this assignment?
2. Would you rate it as easy, moderate, or difficult?
3. Did you work on it mostly alone or did you discuss the problems with others?
4. How deeply do you feel you understand the material it covers (0%–100%)?
5. Any other comments?

Create a pull request from the HW2 branch to main and assign me as the reviewer.
