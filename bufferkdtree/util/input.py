'''
Created on 15.09.2015

@author: Fabian Gieseke
'''

import sys

def ask_question(question):
    """ Helper function to ask user for yes/no 
    input given a particular question.
    
    Parameters
    ----------
    question : str
        The question for which the user shall
        provide the input
    """
    
    yes_set = set(['yes', 'y'])
    no_set = set(['no', 'n'])

    while True:

        sys.stdout.write(question + " ")
        user_input = raw_input().lower()
        
        if user_input in yes_set:
            return True
        
        elif user_input in no_set:
            return False
        
        else:
            
            sys.stdout.write("Only the following answers are allowed: " + unicode(list(yes_set)) + " or " + unicode(list(no_set)) + "\n")
