class RegistryError(Exception):
    def __init__(self, message: str, **kwargs):
        self.context = kwargs
        super().__init__(message)