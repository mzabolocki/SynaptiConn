""" Sub-module for utility functions. """

from .attribute_checks import requires_recording_length, requires_sampling_rate, requires_arguments
from .report import gen_model_results_str
from .warnings import custom_formatwarning