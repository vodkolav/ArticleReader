import os
from Benchmarking.utils import get_path, upd_path

class Case(dict):


    # case: dict

    # config: dict = {}
    # ID: dict = {}
    # tracks: dict = {}
    # summary: dict = {}

    def __init__(self, case: dict):
        #TODO: make checks:
        # case has all required fields
        for k,v in case.items():
            self[k] = v

    # @property
    # def case_index(self):
    #     return self.case["summary"]["case_index"]


    @property
    def config(self) -> dict:
        return self["config"]

    @property
    def ID(self) -> dict:
        return self["ID"]

    @property
    def tracks(self) -> dict:
        return self["tracks"]

    @property
    def summary(self) -> dict:
        return self["summary"]

    @property
    def case_signature(self) -> dict:
        return self.ID["case_signature"]


    @property
    def results(self) -> dict:
        #TODO check if tracks are collected from tele
        return self


    def __eq__(self, other):
        return self.case_signature == other.case_signature


    def __jobj__(self) -> dict:
        return self


    def update_case(self, key, value):
        upd_path(key, self, value)


    def get_path(self, path):
        path = path.strip(".").split(".")
        val = get_path(path, self)
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