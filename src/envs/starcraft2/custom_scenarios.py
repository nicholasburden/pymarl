import numpy as np
import json
from os.path import dirname, join
from functools import partial


def army_placement(ally_centered, rotate, separation):
    if rotate:
        theta = np.random.random() * 2 * np.pi
    else:
        theta = np.pi
    if ally_centered:
        r = separation
        ally_pos = (0, 0)
        enemy_pos = (r * np.cos(theta), r * np.sin(theta))
    else:
        r = separation / 2
        ally_pos = (r * np.cos(theta), r * np.sin(theta))
        enemy_pos = (-ally_pos[0], -ally_pos[1])
    return ally_pos, enemy_pos

def sample_starts(filename, max_types_and_units=True):
    """
    Simply sample starting locations from a saved file
    (Use to replicate static scenarios)
    """
    with open(join(dirname(__file__), 'starts', filename)) as f:
        all_starts = json.load(f)
    start = np.random.choice(all_starts)
    return (start['ally_army'], start['enemy_army'])

def fixed_armies(ally_army, enemy_army, ally_centered=False, rotate=False,
                 separation=10, max_types_and_units=False, jitter=0):
    ally_pos, enemy_pos = army_placement(ally_centered, rotate, separation)
    return ([(num, unit, ally_pos + (np.random.rand(2) - 0.5) * 2 * jitter) for (num, unit) in ally_army],
            [(num, unit, enemy_pos + (np.random.rand(2) - 0.5) * 2 * jitter) for (num, unit) in enemy_army])


def symmetric_armies(unit_types, n_agent_range, ally_centered=False,
                     rotate=False, separation=10, max_types_and_units=False, jitter=0):
    ally_pos, enemy_pos = army_placement(ally_centered, rotate, separation)
    if max_types_and_units:
        n_agents = n_agent_range[1]
    else:
        n_agents = np.random.randint(n_agent_range[0], 1 + n_agent_range[1])
    n_unit_types = len(unit_types)  # number of unique units
    if max_types_and_units:
        partitions = list(range(n_unit_types)) + [n_agents]
    else:
        partitions = [0] + sorted(np.random.randint(n_agents + 1, size=n_unit_types - 1)) + [n_agents]
    ally_army = []
    enemy_army = []
    for unit, s, e in zip(unit_types, partitions[:-1], partitions[1:]):
        if e - s > 0:
            ally_army.append((e - s, unit, ally_pos + (np.random.rand(2) - 0.5) * 2 * jitter))
            enemy_army.append((e - s, unit, enemy_pos + (np.random.rand(2) - 0.5) * 2 * jitter))
    return ally_army, enemy_army

"""
The function in the registry needs to return a tuple of two lists, one for the
ally army and one for the enemy.
Each is of the form [(number, unit_type, pos), ....], where pos is the starting
positiong (relative to center of map) for the corresponding units.
The function will be called on each episode start.
Currently, we only support the same number of agents and enemies each episode.
"""

custom_scenario_registry = {
  "MMM": (partial(fixed_armies,
                  [(1, "Medivac"), (2, "Marauder"), (7, "Marine")],
                  [(1, "Medivac"), (2, "Marauder"), (7, "Marine")],
                  rotate=True),
          {'episode_limit': 150}),
  "MMM_agg": (partial(fixed_armies,
                      [(1, "Medivac"), (2, "Marauder"), (7, "Marine")],
                      [(1, "Medivac"), (2, "Marauder"), (7, "Marine")],
                      rotate=True,
                      ally_centered=True),
              {'episode_limit': 150}),
  # "3s5z": (partial(fixed_armies,
  #                  [(3, "Stalker"), (5, "Zealot")],
  #                  [(3, "Stalker"), (5, "Zealot")],
  #                  rotate=False,
  #                  ally_centered=False,
  #                  separation=14,
  #                  jitter=1),
  #           {'episode_limit': 150}),
  "3s5z": (partial(sample_starts,
                   '3s5z.json'),
           {'episode_limit': 150}),
  "1-5m_symmetric": (partial(symmetric_armies,
                             ('Marine',), (1, 5),
                             rotate=True,
                             ally_centered=True,
                             separation=14),
                     {'episode_limit': 100})
}
