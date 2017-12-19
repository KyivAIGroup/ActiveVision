class EncoderStub(object):
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, value):
        raise NotImplementedError("Stub!")
