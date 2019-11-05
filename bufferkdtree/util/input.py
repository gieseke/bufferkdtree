#
# Copyright (C) 2013-2019 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

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
        if sys.version_info[0] < 3:
            user_input = raw_input().lower()
        else:
            user_input = input().lower()
        
        if user_input in yes_set:
            return True
        
        elif user_input in no_set:
            return False
        
        else:
            
            sys.stdout.write("Only the following answers are allowed: " + str(list(yes_set)) + " or " + str(list(no_set)) + "\n")
