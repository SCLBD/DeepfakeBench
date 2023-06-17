class Registry(object):
    def __init__(self):
        self.data = {}
    
    def register_module(self, module_name=None):
        def _register(cls):
            name = module_name
            if module_name is None:
                name = cls.__name__
            self.data[name] = cls
            return cls
        return _register
    
    def __getitem__(self, key):
        return self.data[key]
    
BACKBONE = Registry()
DETECTOR = Registry()
TRAINER  = Registry()
LOSSFUNC = Registry()
