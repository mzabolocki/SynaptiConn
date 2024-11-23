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

class ConnectionTypeError(SpikeTimesError):
    """Error for if there is a problem with the connection type analysis."""
    
class NoDataError(SpikeTimesError):
    """Error for if there is no data to analyze."""
    
class AnalysisError(SpikeTimesError):
    """ Error for if there is a problem with the analysis. """

class SpikePairError(SpikeTimesError):
    """ Error for if there is a problem with the spike pair analysis. """