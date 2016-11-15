from __future__ import print_function
__author__ = "Harshvardhan Gupta"


win_states = (
[0, 1, 2],
[3, 4, 5],
[6, 7, 8],
[0, 3, 6],
[1, 4, 7],
[2, 5, 8],
[0, 4, 8],
[2, 4, 6])

class Tic(object):


    def __init__(self, selfSide ,positions=[]):
        assert isinstance(positions,list) and (selfSide=="o" or
                                               selfSide=="x")
        self.side = selfSide
        if len(positions) == 0:
            self.positions = [None for i in range(9)]
        else:
            self.positions = positions

    def display(self):
        print("")
        for i in range(0,9):

            if i % 3 == 0:
                print("")

            if(self.positions[i]!=None):
                print(self.positions[i]," ",
                      end="")
            else:
                print ("T"," ",end="")

        print("")

    def getNextPossibleStates(self):
        possibleStates = []
        c=0
        for i in range(9):
            if self.positions[i]==None:
                possibleStates.append(Tic(self.side,self.positions))
                possibleStates[c].positions[i] = self.side
                possibleStates[c].display()





testBoard = ['x','x','o',None,"o",None,None,None,None]
testTic =  Tic("x",testBoard)
testTic.display()
print("that was init")

testTic.getNextPossibleStates()











