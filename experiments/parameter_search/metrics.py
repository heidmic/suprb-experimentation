from suprb import SupRB

__all__ = ['fitness']


def fitness(model: SupRB):
    return model.elitist_.fitness_
