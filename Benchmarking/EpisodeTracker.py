import numpy as np
from copy import deepcopy
import time


class EpisodeTracker:

    def __init__(self, sampling_type = "interval_sec", sampling_value = 0.1, **kwargs ):
        self.episodes = []
        # conf = case["tracks"]["episodes"], for ex. 
        self.exception = None
        self._sampling_type = sampling_type
        self._sampling_value = sampling_value
        self.episodes.clear()  # Clear previous logs
        
        # add first empty record to split the 
        # different cases on the graphs   
        self.episodes.append({
            "start_time": self.now(),
            "end_time": self.now()})
        
        self.config_scheduling()
        self.record = {}

    @property
    def total_episodes(self):
        return self.tot_episodes


    @total_episodes.setter
    def total_episodes(self, value):
        self.tot_episodes = value


    @property
    def sampling_type(self):
        return self._sampling_type


    @property
    def sampling_value(self):
        return self._sampling_value

    def now(self):
        # TODO: variable format
        return time.time()

    def config_scheduling(self, val = -1):
        # sampling_type:  interval_sec, interval_episodes, total_samples
        # sampling_value:          0.1,                 4,           100

        match self.sampling_type:

            case "interval_episodes":
                self.last_sample = val

            case "interval_sec":
                self.last_sample = val

            case "total_samples":
                tot = self.total_episodes # 2342
                value = self.sampling_value # total_samples = 100

                if  tot < value:
                    value = tot
                
                self.samplePoints = np.int64(np.linspace(0, tot, num = value ))
                invl = np.round(value/tot, decimals=2)
                print("tracking and reporting once every", invl, "episodes")


    def time_to_record(self, i_episode):
        match self.sampling_type:

            case "interval_episodes":
                if i_episode >= self.last_sample + self.sampling_value:
                    self.last_sample = i_episode
                    return True

            case "interval_sec":
                timE = time.time()
                if timE >= self.last_sample + self.sampling_value:
                    self.last_sample = timE
                    return True

            case "total_samples":
                # total_samples
                if i_episode in self.samplePoints:
                    return True

        return False


    def attach_to(self, episode_func, summary_func):
        # self.runner_function = episode_func
        # self.summarizer_function = summary_func

        def record_episode_wrapper(*args, **kwargs):
            
            # if "i_batch" in kwargs: # TODO: check this at the time of attachment?
            #     i_episode =  kwargs["i_batch"]
            # else: 
            #     raise AttributeError("must include i_episode")

            i_episode = args[0]

            
            try:
                allowed = self.time_to_record(i_episode)
                if allowed:
                    self.record["i"] = i_episode
                    self.record["start_time"] = self.now()
                    output = episode_func(*args, **kwargs)  # Run the original forward pass
                    self.record["end_time"] = self.now()

                else:
                    output = episode_func(*args, **kwargs)  # Run the original forward pass

            except Exception as e:
                print("episode_func failed with exc: ", str(e))
                self.exception = str(e)
                output = None
            finally:
                if allowed:
                    data = summary_func(*args, **kwargs)
                    self.record.update(deepcopy(data))
                    self.episodes.append(self.record)
                    self.record = {}
                    allowed = False

            return output
        return record_episode_wrapper


    def summarize(self):

        if len(self.episodes)>1:
            # data = pd.DataFrame(self.memory_log)
            # data['time'] = pd.to_datetime(data['time'], unit='s')
            # dur = (data.time.max() - data.time.min()).total_seconds()
            a = self.episodes[1]['start_time']
            b = self.episodes[-1]['end_time']
            dur = b-a
            #  data['memory'].max()
            #dur = str(dur.microseconds/1e6)
        else:
            dur=0

        res = {
            "sampling_type": self._sampling_type,
            "sampling_value": self.sampling_value,
            "run_time_sec": dur,
            "data": self.episodes,
            "exceptions": self.exception,
        }
        return res