'''
Created on 18.11.2015

@author: Fabian Gieseke
'''

import multiprocessing

def wrapped_task(proc_num, task, args, kwargs, return_dict):

    return_dict[proc_num] = task(*args, **kwargs)
            
def start_via_single_thread(task, args, kwargs):
    

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    proc_num = 0
    proc = multiprocessing.Process(target=wrapped_task, args=(proc_num, task, args, kwargs, return_dict))
            
    proc.daemon = False
    proc.start()

    proc.join()

    return return_dict[proc_num]   