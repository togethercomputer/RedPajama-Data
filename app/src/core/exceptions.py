class S3ReadError(Exception):
    def __init__(self, message):
        super().__init__(message)


class S3WriteError(Exception):
    def __init__(self, message):
        super().__init__(message)


class LocalReadError(Exception):
    def __init__(self, message):
        super().__init__(message)


class UnknownReadError(Exception):
    def __init__(self, message):
        super().__init__(message)
