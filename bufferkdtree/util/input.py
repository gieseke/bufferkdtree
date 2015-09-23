'''
Created on 15.09.2015

@author: fgieseke
'''

import sys

def ask_question(question, default='yes'):
    """ Helper function to ask user for yes/no 
    input given a particular question.
    
    Parameters
    ----------
    question : str
        The question for which the user shall
        provide the input
    default : str
        The default answer
        
    """
    yes_set = set(['yes', 'y', ''])
    no_set = set(['no', 'n'])

    if default is None:
        default_answer = " [y/n] "
    elif default == "yes":
        default_answer = " [Y/n] "
    elif default == "no":
        default_answer = " [y/N] "
    else:
        raise ValueError("Invalid default default_answer '%s'" % default)

    while True:

        sys.stdout.write(question + default_answer)
        user_input = raw_input().lower()
        
        if user_input in yes_set:
            return True
        elif user_input in no_set:
            return False
        else:
            sys.stdout.write("Only the followed answers are allowed: " + unicode(yes_set) + " and " + unicode(no_set) + "\n")
