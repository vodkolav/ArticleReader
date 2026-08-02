import os
from Benchmarking.utils import get_path, upd_path

class Case:


    case: dict

    # config: dict = {}
    # ID: dict = {}
    # tracks: dict = {}
    # summary: dict = {}

    def __init__(self, case: dict):
        #TODO: make checks:
        # case has all required fields
        self.case = case

    # @property
    # def case_index(self):
    #     return self.case["summary"]["case_index"]


    @property
    def config(self):
        return self.case["config"]

    @property
    def ID(self):
        return self.case["ID"]

    @property
    def tracks(self):
        return self.case["tracks"]

    @property
    def summary(self):
        return self.case["summary"]

    @property
    def case_signature(self):
        return self.ID["case_signature"]


    @property
    def results(self):
        #TODO check if tracks are collected from tele
        return self.case




    def __eq__(self, other):
        return self.case_signature == other.case_signature

    
    def __pos__(self):
        return True


    def update_case(self, key, value):
        # TODO: kinda ugly, should update in place, 
        # not overwrite the whole dict
        self.case = upd_path(key, self.case, value)

    def get_path(self, path):
        path = path.strip(".").split(".")
        val = get_path(path, self.case)
        return val


    def case_filename(self, field = "case_id"):
        opts = ["case_id", "case_signature"]
        if field not in opts:
            raise ValueError("Possible values for field are: " + str(opts))

        case_id = self.ID[field]
        experiment_id = self.ID["experiment_id"]
        exp_dir = os.path.join(self.ID['output_root'], experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        pth = os.path.join(exp_dir, case_id)
        return pth