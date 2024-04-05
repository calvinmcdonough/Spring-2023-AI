import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """
    #Add more comments for checkin

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        #this will be a list of all of the sentences that have the value equal to the elements in the set. 

        return self.cells

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        safe = set()
        if self.count == 0:
            for i in self.cells:
                set.add(i)

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        

        
        
        return None

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
        
        return None


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        #newSentence = Sentence() # initiate the sentence that will be added
        #1)
        self.moves_made.add(cell)
        self.mark_safe(cell) 

        #2)

        sentenceCells = self.sorounding_cells(cell)
        
        

           
        #3) this adds a new sentence to the ais knowledge base
        
        newSentence = Sentence(sentenceCells, count) 
        
        self.knowledge.append(newSentence)
        
        

        #4) update the knowledge base 

        # handle the case where the count of the current cell is 0
        if count == 0:
            for celll in sentenceCells:
                self.mark_safe(celll)
            
        
        # look at the count of the cells in the centence and see if they are known to be mines
        for sentence in self.knowledge:
            if len(sentence.cells) == sentence.count: # this looks to see if the number of cells in the sentence are equall to the count
                for sell in sentence.cells: # hehehehe sell this marks the cells in the sentence as mines when the sentence equalls the count 
                    self.mark_mine(sell)
        
        # update the sentence by taking out the known mines and also subtracting from the count.
        for sentence in self.knowledge:
            for mine in self.mines:
                if (mine in sentence.cells) == True:
                   
                    sentence.cells.remove(mine) #this removes the mine from the sentence
                    sentence.count = sentence.count-1 # this subtracts the count of the sentence to 

        #lets deal with the subsets of the cells and see how that can help us 
        # this is definetly not the most efficient way to do this but its midnight on thursday right now so I dont have the brain power to optomize this
        for sentence in self.knowledge: # this iterates though all of the sentences in knowledge this will be the sub set 
            for sentence2 in self.knowledge: # this looks through all the sentences in knoledge too 
                if sentence.cells.issubset(sentence2.cells) == True and (sentence.__eq__(sentence2)) == False: # the way im doing this i think in need to make sure that they are not equall because then they would delet eacchother 
                   
                    sentence2.count = sentence2.count - sentence.count 
                  
                    for sell in sentence.cells:
                        if (sell in sentence2.cells) == True: # if the cell is in the cells then the cell is removed from the sentence
                            sentence2.cells.remove(sell)

                    

                    

        
        # iterate through all of the sentences after the updates to see if we can mark any new cells as sate 
        newSafes = set() #this is dumb 
        for sentence in self.knowledge: # iterates throught the sentences
            if sentence.count == 0: # if the count of the sentence is equal to 0 then we can update the 
                for sell in sentence.cells:
                    
                    newSafes.add(sell) # this is why it is dumb because if you mark safe here then you shrink the size of the sentence that you are iterating through in the mark safe in the sentence class
        for sell in newSafes:# so basically you have to update the safe cells after you iterate throught the all the sentences and find the new safes
            self.mark_safe(sell)

        # this is for the print
        for i in self.knowledge:
            print(i)

        # now we are going to loop through the moves that we have made and erase them 
        for sentence in self.knowledge:
            for move in self.moves_made:
                if (move in sentence.cells) == True:
                    sentence.cells.remove(move)
        
        # this is for the print a second time 
        for i in self.knowledge:
            print(i)

        return None


    def sorounding_cells(self, cell):
        '''
        This method returns a set of all of the sorrounding cells for a passed in cell
        '''
        i,j = cell
        sentenceCells = set()
        
        if i-1>=0:
            newCell = (i-1,j) #left cell
            sentenceCells.add(newCell)
        if i-1>=0 and j-1>=0:
            newCell = (i-1,j-1)# up and to the left
            sentenceCells.add(newCell)
        if i-1 >= 0 and j+1 < self.width:
            newCell = (i-1,j+1) # down and to the left
            sentenceCells.add(newCell)
        if i+1 < self.height:
            newCell = (i+1,j) # right cell  
            sentenceCells.add(newCell)
        if i+1 < self.height and j-1>= 0:
            newCell = (i+1,j-1) # right and up 
            sentenceCells.add(newCell)
        if i+1 < self.height and j+1< self.width:
            newCell = (i+1,j+1) # right and down
            sentenceCells.add(newCell)
        if j+1< self.width:
            newCell = (i,j+1) # down cell
            sentenceCells.add(newCell)
        if j-1>= 0:
            newCell = (i,j-1) # above cell
            sentenceCells.add(newCell)
        return sentenceCells

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell1 in self.safes:
            if (cell1 in self.moves_made) == False:
                return cell1
        return None 

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if len(self.moves_made) ==0:
            return (2,1)
        
        for i in range(self.height-1): # loopes through the board
            for j in range(self.width-1):
                move = (i,j)
                if (move in self.moves_made) == False and (move in self.mines) == False: # checks to see if the move has been played
                    return move
        return None
        '''
        store = None 
        percent = 2
        for sentence in self.knowledge:
            if len(sentence.cells) >0:
                num = sentence.count/len(sentence.cells)
                if num < percent:
                    for sell in sentence.cells:
                        if (sell in self.moves_made) == False and (sell in self.mines) == False:
                            print(sell, " this is the move that is supposed to be made " ,num , " this is the percent")
                            print(self.mines, " these are the known mines")
                            store = sell
                            percent = num
                            
        return store
                 '''   
        
