class Config(dict):
    def __init__(self, seq=None, **kwargs):
        super().__init__(seq, **kwargs)
        self._to_config()

    def _to_config(self):
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, item):
        if item in self.keys():
            return self[item]
        else:
            raise AttributeError(f'{self} object has no attribute {item}')

    def __setattr__(self, key, value):
        if key in self.keys():
            self[key] = value
        else:
            raise AttributeError(f'{self} object has no attribute {key}')
