class SRLError(Exception):
    def __repr__(self):
        return "%s: An unspecified SRL error has occurred; %s" % (self.__class__.__name__, self.args)


class UndefinedError(SRLError):
    pass


class NotSupportedError(SRLError):
    pass


class TFLayerError(SRLError):
    pass


class DistributionError(SRLError):
    pass
