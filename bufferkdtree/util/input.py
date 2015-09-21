import sys

def question(q, default='yes'):

    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:

        sys.stdout.write(q + prompt)
        choice = raw_input().lower()
        if choice in yes:
           return True
        elif choice in no:
           return False
        else:
           sys.stdout.write("Allowed answers: 'yes' or 'no'\n")
