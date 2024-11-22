"""warnings.py

Modules for custom warnings.
"""


def custom_formatwarning(msg, *args, **kwargs):
    """ Custom format for warnings. """
    return f"{msg}\n"