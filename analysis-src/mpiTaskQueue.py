''' MPI task queue abstraction that runs a service.

Ben Porebski, MRC Laboratory of Molecular Biology, 18/5/19
'''

from mpi4py import MPI
import time, logging

comm = MPI.COMM_WORLD
status = MPI.Status()

timeout = 300

class TaskQueue:
    '''Task queue class.'''

    def taskWorkerLoop(self):
        '''Task worker loop'''

        tasks = []
        results = []

        print("Worker started on %s" % (comm.rank))
        t_last_task = time.time()
        while True:
            while not comm.Iprobe():
                if tasks != []:
                    try:
                        if tasks[0] == -1:
                            comm.Abort(1)
                            exit()
                    except:
                        None
                    ## The received data comes as [task_src, [img batch data]]
                    task_src = tasks[0][0]
                    task = tasks[0][1]
                    comm.send(self.model.predict(task, batch_size=128), dest=task_src)
                    tasks.pop(0)
                    t_last_task = time.time()
                else:
                    t_cur = time.time()
                    t_since_last = t_cur - t_last_task
                    if t_since_last > timeout: ## If no task in the last 300s, terminate.
                        print("Worker %s has not received a task in > %d seconds. Terminating." % (comm.rank, timeout))
                        comm.Abort(1)
                        exit()

            tasks.append(comm.recv())
            t_last_task = time.time()




    def submitTask(self, task):
        # print "Distributing task: %s" % (taskList)

        procToSend = 0
        comm.send(task, dest=procToSend)
        results = comm.recv(source=MPI.ANY_SOURCE)

        return results


    def terminate(self):
        '''Terminates the task queue.'''
        for rank in range(1, comm.size):
            comm.send(-1, dest=rank)

    def __init__(self, model):
        if comm.rank == 0:
            self.model = model
            self.taskWorkerLoop()
