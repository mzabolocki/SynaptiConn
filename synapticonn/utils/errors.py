""" Custom error definitions. """

class SpikeTimesError(Exception):
    """Base class for custom errors."""

class DataError(SpikeTimesError):
    """Error for if there is a problem with the data."""

class SamplingRateError(SpikeTimesError):
    """Error for if there is a problem with the sampling rate."""

class RecordingLengthError(SpikeTimesError):
    """Error for if there is a problem with the recording length."""

class PlottingError(SpikeTimesError):
    """Error for if there is a problem with plotting."""