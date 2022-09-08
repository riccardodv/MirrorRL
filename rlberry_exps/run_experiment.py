"""
To run the experiment:

$ python run.py experiment.yaml

To see more options:

$ python run.py -h
"""

from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers
#import pdb



def main():

    multimanagers = MultipleManagers()

    for agent_manager in experiment_generator():
        multimanagers.append(agent_manager)

    # Alternatively:
    # agent_manager.fit()
    # agent_manager.save()

    #pdb.set_trace()
    multimanagers.run()
    multimanagers.save()


if __name__ == '__main__':
    main()
