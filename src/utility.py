import datetime


class Timer():

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = datetime.datetime.now()

    def stop(self):
        self.end_time = datetime.datetime.now()
        print("Time taken: %s" % (self.end_time - self.start_time))