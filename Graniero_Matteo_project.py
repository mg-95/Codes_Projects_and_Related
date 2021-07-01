######## Problem 1 ##########

class Node:

    def __init__(self, element):
        self.element = element
        self.next = None

    def getElement(self):
        return self.element

    def getNext(self):
        return self.next

    def setNext(self, n):
        self.next = n

    def __str__(self):
        return str(self.element)

class Floyd:

    def __init__(self):
        self.head = None

    def detectLoop(self):
        slow_tortoise = fast_hare = self.head

        while(slow_tortoise and fast_hare and fast_hare.next):
            slow_tortoise = slow_tortoise.next
            fast_hare = fast_hare.next.next
         
            if slow_tortoise == fast_hare:
                self.removeLoop(slow_tortoise)

                return 'Loop present'

        return 'No loop present'

    def removeLoop(self, loop):
        ptr1 = self.head

        while(1):
            ptr2 = loop

            while(ptr2.next != loop and ptr2.next != ptr1):
                ptr2 = ptr2.next

            if ptr2.next == ptr1:
                break

            ptr1 = ptr1.next

        ptr2.next = None


    def __str__(self):
        res = '('
        cursor = self.head
        
        while cursor:
            res = res + str(cursor.getElement())
            if cursor != self.tail:
                res = res + ',' 
            cursor = cursor.getNext()
            
        res = res + ')'
        return res 

######### Problem 2 ##########

class RoundRobin():
    
    def __init__(self,quantum):
        self.pArrival=[]
        self.pTime=[]
        self.quantum=quantum
        self.time=0

    def addProcess(self,arrival,exeTime):
        
        if len(self.pArrival)==0:
            self.pArrival.append(arrival)
            self.pTime.append(exeTime)
            
        else:
            i=0
            while(i<len(self.pArrival)):
                if arrival<self.pArrival[i]:
                    break
                i+=1
            self.pArrival.insert(i,arrival)
            self.pTime.append(exeTime)

    def ProcessAll(self):
        
        print(self.pArrival,self.pTime)
        
        while(self.pTime):
            
            if self.pTime[0]>=self.quantum:
                a=self.pTime.pop(0)
                if a>0:
                    self.pTime.append(a-self.quantum)
                    
            elif self.pTime[0]<self.quantum:
                a=self.pTime.pop(0)
                rest=self.quantum-a
                while(rest and self.pTime):
                    if rest<=self.pTime[0]:
                        self.pTime[0]=self.pTime[0]-rest
                        rest=0
                    else:
                        rest=rest-self.pTime.pop(0)
                        
            self.time+=self.quantum
            print(self.pTime)
            
        self.pArrival=[]
        
        print(self.time,self.pArrival)
        
if __name__=="__main__":

    ProcessTable = [['P0',0,250],
                   ["P1",50,170],
                   ["P2",130,75],
                   ["P3",190,100],
                   ["P4",210,130],
                   ["P5",350,50]]
    RR=RoundRobin(100)
    
    for i in range(len(ProcessTable)):
        RR.addProcess(ProcessTable[i][1],ProcessTable[i][2])
        
    RR.ProcessAll()
