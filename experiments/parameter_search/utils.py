from functools import wraps

from sklearn.utils import Bunch


def param_space(prefix=''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            bunch = Bunch()
            func(params=bunch, *args, **kwargs)
            return {prefix + ('__' if prefix != '' else '') + key: value for key, value in bunch.items()}

        return wrapper

    return decorator


def individual_optimizer_space(func):
    return param_space('individual_optimizer')(func)
