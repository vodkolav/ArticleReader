


class Sensor:

    def __init__(self):
        pass

    def attach_to(self, **kwargs):
        pass

    def results(self, **kwargs):
        pass

    def summarize(self):
        pass


        #TODO: implement different forms of scheduling reports
        # - total episodes to report (requires how many total episodes will be)
        # - once every x episodes (frequency)
        # - time-based
        # - on demand: whenever something happens (log)
        
        # basically it already exists in EpisodeTracker. 
        # need to check if it makes sense for every sensor. 
        # Then it can be moved to this base class

