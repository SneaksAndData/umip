"""Optimization specific exceptions."""

class OptimizationException(Exception):
    """A general optimization exception"""


class InfeasibleException(OptimizationException):
    """An exception for when the model is infeasible"""


class AbnormalException(OptimizationException):
    """An exception for when the model is abnormal"""
