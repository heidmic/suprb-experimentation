from suprb2 import SupRB2

__all__ = ['fitness']


def fitness(model: SupRB2):
    return model.elitist_.fitness_
