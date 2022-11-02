"""Base Class and PIPELINE register for the preprocessing pipeline scripts."""

PIPELINE = {}

class _Preprocess(object):
    
    def __init__(self, step_cfgs):
        self.process_cfgs = step_cfgs.get("process")
        self.calc_cfgs = step_cfgs.get("calc")
        self.save_cfgs = step_cfgs.get("save")
        self.select_cfgs = step_cfgs.get("select")
    
    def process(self, aman):
        if self.process_cfgs is None:
            return
        raise NotImplementedError
        
    def calc_and_save(self, aman, proc_aman):
        if self.calc_cfgs is None:
            return
        raise NotImplementedError
    
    def save(self, proc_aman, *args):
        if self.save_cfgs is None:
            return
        raise NotImplementedError
        
    def select(self, meta):
        if self.select_cfgs is None:
            return meta
        raise NotImplementedError
    
    @staticmethod
    def register(name, process_class):
        if PIPELINE.get(name) is None:
            PIPELINE[name] = process_class



