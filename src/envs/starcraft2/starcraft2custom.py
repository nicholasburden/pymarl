from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.dict2namedtuple import convert
from ..multiagentenv import MultiAgentEnv
from .custom_scenarios import custom_scenario_registry

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import torch as th
from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from pysc2.lib.units import get_unit_type

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb
from .map_params import get_map_params, map_present
import sys
import os

difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

def get_unit_type_by_name(name):
    for race in (Neutral, Protoss, Terran, Zerg):
        unit = getattr(race, name, None)
        if unit is not None:
            return unit
    # raise ValueError(f"Bad unit type {name}")  # this gives a syntax error for some reason


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class StarCraft2CustomEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(self, **kwargs):

        args = kwargs
        if isinstance(args, dict):
            args = convert(args)

        self.obs_timestep_number = False

        self.state_timestep_number = False
        # Read arguments
        self.map_name = args.map_name
        assert map_present(self.map_name), \
            "map {} not in map registry! please add.".format(self.map_name)
        map_params = convert(get_map_params(self.map_name))
        self.n_agents = map_params.n_agents
        self.n_enemies = map_params.n_enemies
        self.episode_limit = map_params.limit
        self._move_amount = args.move_amount
        self._step_mul = args.step_mul
        self.difficulty = args.difficulty

        # Observations and state
        self.obs_own_health = args.obs_own_health
        self.obs_all_health = args.obs_all_health
        self.obs_instead_of_state = args.obs_instead_of_state
        self.obs_last_action = args.obs_last_action
        self.obs_pathing_grid = args.obs_pathing_grid
        self.obs_terrain_height = args.obs_terrain_height
        self.state_last_action = args.state_last_action
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        # Rewards args
        self.reward_sparse = args.reward_sparse
        self.reward_only_positive = args.reward_only_positive
        self.reward_negative_scale = args.reward_negative_scale
        self.reward_death_value = args.reward_death_value
        self.reward_win = args.reward_win
        self.reward_defeat = args.reward_defeat
        self.reward_scale = args.reward_scale
        self.reward_scale_rate = args.reward_scale_rate

        # Other
        self.continuing_episode = args.continuing_episode
        self.seed = args.seed
        self.heuristic = args.heuristic_ai
        self.window_size = (2560, 1600)
        self.save_replay_prefix = args.replay_prefix
        self.restrict_actions = True  # args.restrict_actions

        # For sanity check
        self.debug_inputs = False
        self.debug_rewards = False

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.action_representation = getattr(args, "action_representation", "original")

        # Configuration related to obs featurisation
        self.obs_id_encoding = getattr(args, "obs_id_encoding", "original")  # one of: original, metamix
        self.obs_decoder = getattr(args, "obs_decoder", None)  # None: flatten output!
        self.obs_decode_on_the_fly = getattr(args, "obs_decode_on_the_fly", True)
        self.obs_grid_shape = list(map(int, getattr(args, "obs_grid_shape", "1x1").split("x")))  # (width, height)
        self.obs_resolve_multiple_occupancy = getattr(args, "obs_resolve_multiple_occupancy", False)
        self.obs_grid_rasterise = getattr(args, "obs_grid_rasterise", False)

        self.obs_grid_rasterise_debug = getattr(args, "obs_grid_rasterise_debug", False)
        self.obs_resolve_multiple_occupancy_debug = getattr(args, "obs_resolve_multiple_occupancy_debug", False)
        assert not (self.obs_resolve_multiple_occupancy and (self.obs_grid_shape is None)), "obs_grid_shape required!"

        if self.obs_decoder is not None:
            self.obs_id_encoding = "metamix"
            self.rasterise = True
            self.obs_resolve_multiple_occupancy = True

        # Finalize action setup
        self.n_actions = self.n_actions_no_attack + self.n_enemies
        if self.action_representation == "output_grid":
            self.n_actions_encoding = self.obs_grid_shape[0] * self.obs_grid_shape[1] + self.n_actions_no_attack
        elif self.action_representation == "input_grid":
            self.n_actions_encoding = self.obs_grid_shape[0] * self.obs_grid_shape[1] + self.n_actions_no_attack
        elif self.action_representation == "input_xy":
            self.n_actions_encoding = self.obs_grid_shape[0] + self.obs_grid_shape[1] + self.n_actions_no_attack
        else:
            self.n_actions_encoding = self.n_actions

        self.n_available_actions = self.n_actions_no_attack + self.n_enemies

        # Map info
        self._agent_race = map_params.a_race
        self._bot_race = map_params.b_race
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0

        #  NOTE: This means we ALWAYS have a unit type bit! really important for Metamix!
        self.unit_type_bits = map_params.unit_type_bits
        self.map_type = map_params.map_type

        if sys.platform == 'linux':
            os.environ['SC2PATH'] = os.path.join(os.getcwd(), "3rdparty", 'StarCraftII')
            self.game_version = args.game_version
        else:
            # Can be derived automatically
            self.game_version = None

        # Launch the game
        self._launch()

        self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win
        self._game_info = self._controller.game_info()
        self._map_info = self._game_info.start_raw
        self.map_x = self._map_info.map_size.x
        self.map_y = self._map_info.map_size.y
        self.map_play_area_min = self._map_info.playable_area.p0
        self.map_play_area_max = self._map_info.playable_area.p1
        self.max_distance_x = self.map_play_area_max.x - self.map_play_area_min.x
        self.max_distance_y = self.map_play_area_max.y - self.map_play_area_min.y
        self.terrain_height = np.flip(
            np.transpose(np.array(list(self._map_info.terrain_height.data)).reshape(self.map_x, self.map_y)), 1)
        self.pathing_grid = np.flip(
            np.transpose(np.array(list(self._map_info.pathing_grid.data)).reshape(self.map_x, self.map_y)), 1)

        self._episode_count = 0
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        self.last_stats = None

        # Calculate feature sizes

        obs_items = {"ally": [], "enemy": [], "own": []}
        obs_items["ally"].append(("visible", 1,))
        obs_items["ally"].append(("distance_sc2", 1,))
        obs_items["ally"].append(("x_sc2", 1,))
        obs_items["ally"].append(("y_sc2", 1,))

        obs_items["enemy"].append(("attackable", 1,))
        obs_items["enemy"].append(("distance_sc2", 1,))
        obs_items["enemy"].append(("x_sc2", 1,))
        obs_items["enemy"].append(("y_sc2", 1,))

        if self.unit_type_bits:
            obs_items["ally"].append(("unit_local", self.unit_type_bits))
            obs_items["enemy"].append(("unit_local", self.unit_type_bits))

        if self.obs_id_encoding == "metamix":
            obs_items["ally"].append(("unit_global", 1))
            obs_items["enemy"].append(("unit_global", 1))

        if self.obs_id_encoding == "metamix":
            obs_items["enemy"].append(("visible", 1,))

        if self.obs_all_health:
            obs_items["ally"].append(("health", 1,))
            obs_items["enemy"].append(("health", 1,))
            if self.shield_bits_ally:
                obs_items["ally"].append(("shield", 1,))
                obs_items["enemy"].append(("shield", 1,))

        if self.obs_last_action:
            obs_items["ally"].append(("last_action", 1,))

        obs_items["own"].append(("unit_local", self.unit_type_bits))

        if self.obs_id_encoding == "metamix":
            obs_items["own"].append(("unit_global", 1,))

        if self.obs_own_health:
            obs_items["own"].append(("health", 1,))
            if self.shield_bits_ally:
                obs_items["own"].append(("health", self.shield_bits_ally))

        if (self.obs_grid_rasterise and not self.obs_grid_rasterise_debug) \
                or (self.obs_resolve_multiple_occupancy and not self.obs_resolve_multiple_occupancy_debug):
            obs_items["ally"].append(("distance_grid", 1,))
            obs_items["enemy"].append(("distance_grid", 1,))
            obs_items["ally"].append(("x_grid", 1,))
            obs_items["enemy"].append(("x_grid", 1,))
            obs_items["ally"].append(("y_grid", 1,))
            obs_items["enemy"].append(("y_grid", 1,))
            obs_items["ally"].append(("distance_sc2_raster", 1,))
            obs_items["enemy"].append(("distance_sc2_raster", 1,))
            obs_items["ally"].append(("x_sc2_raster", 1,))
            obs_items["enemy"].append(("x_sc2_raster", 1,))
            obs_items["ally"].append(("y_sc2_raster", 1,))
            obs_items["enemy"].append(("y_sc2_raster", 1,))
            obs_items["ally"].append(("id", 1,))
            obs_items["enemy"].append(("id", 1,))

        self.nf_al = sum([x[1] for x in obs_items["ally"]])
        self.nf_en = sum([x[1] for x in obs_items["enemy"]])
        self.nf_own = sum([x[1] for x in obs_items["own"]])

        # now calculate feature indices
        def _calc_feat_idxs(lst):
            ctr = 0
            dct = {}
            for l in lst:
                name = l[0]
                length = l[1]
                dct[name] = (ctr, ctr + length)
                ctr += length
            return dct

        self.move_feats_len = self.n_actions
        self.move_feats_size = self.move_feats_len
        self.enemy_feats_size = self.n_enemies * self.nf_en
        self.ally_feats_size = (self.n_agents - 1) * self.nf_al
        self.own_feats_size = self.nf_own

        self.obs_size_base = self.ally_feats_size + self.enemy_feats_size + self.move_feats_size + self.own_feats_size
        self.obs_size = self.obs_size_base

        idxs = {"ally": _calc_feat_idxs(obs_items["ally"]),
                "enemy": _calc_feat_idxs(obs_items["enemy"]),
                "own": _calc_feat_idxs(obs_items["own"])}

        # create obs access functions
        from functools import partial
        self.obs_get = partial(StarCraft2CustomEnv._obs_get,
                               idxs=idxs)

        self.obs_set = partial(StarCraft2CustomEnv._obs_set,
                               idxs=idxs)

        self.to_grid_coords = partial(StarCraft2CustomEnv._to_grid_coords,
                                      obs_grid_shape=self.obs_grid_shape,
                                      )

        self.from_grid_coords = partial(StarCraft2CustomEnv._from_grid_coords,
                                        obs_grid_shape=self.obs_grid_shape,
                                        )

        self.rasterise_grid = partial(StarCraft2CustomEnv._grid_rasterise,
                                      obs_grid_shape=self.obs_grid_shape,
                                      obs_get=self.obs_get,
                                      obs_set=self.obs_set,
                                      to_grid_coords=self.to_grid_coords,
                                      from_grid_coords=self.from_grid_coords,
                                      debug=self.obs_grid_rasterise_debug
                                      )

        self.resolve_multiple_occupancy = partial(StarCraft2CustomEnv._multiple_occupancy,
                                                  obs_grid_shape=self.obs_grid_shape,
                                                  obs_get=self.obs_get,
                                                  obs_set=self.obs_set,
                                                  to_grid_coords=self.to_grid_coords,
                                                  from_grid_coords=self.from_grid_coords,
                                                  debug=self.obs_resolve_multiple_occupancy_debug
                                                  )

        self.obs_explode = partial(StarCraft2CustomEnv._obs_explode,
                                   ally_feats_size=self.ally_feats_size,
                                   enemy_feats_size=self.enemy_feats_size,
                                   move_feats_len=self.move_feats_len,
                                   n_allies=self.n_agents - 1,
                                   n_enemies=self.n_enemies,
                                   nf_al=self.nf_al,
                                   nf_en=self.nf_en, )

        self.create_channels = partial(StarCraft2CustomEnv._create_channels,
                                       n_allies=self.n_agents - 1,
                                       n_enemies=self.n_enemies,
                                       obs_grid_shape=self.obs_grid_shape,
                                       obs_get=self.obs_get,
                                       obs_set=self.obs_set,
                                       to_grid_coords=self.to_grid_coords,
                                       from_grid_coords=self.from_grid_coords,
                                       obs_explode=self.obs_explode)
        return

    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)

        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        self._sc2_proc = self._run_config.start(window_size=self.window_size)
        self._controller = self._sc2_proc.controller
        self._bot_controller = self._sc2_proc.controller

        # Request to create the game
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self.seed)
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        self.map_center = (self.map_x//2,self.map_y//2)

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                    self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data)).reshape(
                self.map_x, self.map_y)), 1) / 255

    def _calc_distance_mtx(self):
        # Calculate distances of all agents to all agents and enemies (for visibility calculations)
        dist_mtx = 1000 * np.ones((self.n_agents, self.n_agents + self.n_enemies))
        for i in range(self.n_agents):
            for j in range(self.n_agents + self.n_enemies):
                if j < i:
                    continue
                elif j == i:
                    dist_mtx[i, j] = 0.0
                else:
                    unit_a = self.agents[i]
                    if j >= self.n_agents:
                        unit_b = self.enemies[j - self.n_agents]
                    else:
                        unit_b = self.agents[j]
                    if unit_a.health > 0 and unit_b.health > 0:
                        dist = self.distance(unit_a.pos.x, unit_a.pos.y,
                                             unit_b.pos.x, unit_b.pos.y)
                        dist_mtx[i, j] = dist
                        if j < self.n_agents:
                            dist_mtx[j, i] = dist
        self.dist_mtx = dist_mtx

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()

        try:
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            self.init_units()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        self._calc_distance_mtx()

        if self.entity_scheme:
            return self.get_entities(), self.get_masks()
        return self.get_obs(), self.get_state()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def try_controller_step(self, fn=lambda: None, n_steps=1):
        try:
            fn()
            self._controller.step(n_steps)
            self._obs = self._controller.observe()
            return True
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            self._obs = self._controller.observe()
            return False

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions = [int(a) for a in actions[:self.n_agents]]

        self.last_action = np.eye(self.n_actions)[np.array(actions)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions):
            if not self.heuristic_ai:
                agent_action = self.get_agent_action(a_id, action)
            else:
                agent_action = self.get_agent_action_heuristic(a_id, action)
            if agent_action:
                sc_actions.append(agent_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        step_success = self.try_controller_step(lambda: self._controller.actions(req_actions), self._step_mul)
        if not step_success:
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()
        self._calc_distance_mtx()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        return reward, terminated, info

    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        else:
            # attack/heal units that are in range
            target_tag = action - self.n_actions_no_attack
            if unit.unit_type == Terran.Medivac:
                target_id = np.where(self.ally_tags == target_tag)[0].item()
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_id = np.where(self.enemy_tags == target_tag)[0].item()
                target_unit = self.enemies[target_id]
                action_name = "attack"

            action_id = actions[action_name]
            target_unit_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_unit_tag,
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == Terran.Medivac:
            if (target is None or self.agents[target].health == 0 or
                self.agents[target].health == self.agents[target].health_max):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == Terran.Medivac:
                        continue
                    if (al_unit.health != 0 and
                        al_unit.health != al_unit.health_max):
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             al_unit.pos.x, al_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (unit.unit_type == Terran.Marauder and
                        e_unit.unit_type == Terran.Medivac):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        cmd = r_pb.ActionRawUnitCommand(
            ability_id=action_id,
            target_unit_tag=target_tag,
            unit_tags=[tag],
            queue_command=False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health +
                    self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health +
                    self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        """Returns the shooting range for an agent."""
        return 6

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return self._sight_range

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        logging.info("Replay saved at: %s" % replay_path)

    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return (0 <= x < self.map_x and 0 <= y < self.map_y)

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals

    def get_masks(self):
        """
        Returns:
        1) per agent observability mask over all entities (unoberserved = 1, else 0)
        3) mask of inactive entities (including enemies) over all possible entities
        """
        sight_range = np.array(
            [self.unit_sight_range(a_i)
             for a_i in range(self.n_agents)]).reshape(-1, 1)
        obs_mask = (self.dist_mtx > sight_range).astype(np.uint8)
        obs_mask_padded = np.ones((self.max_n_agents,
                                   self.max_n_agents + self.max_n_enemies),
                                  dtype=np.uint8)
        obs_mask_padded[:self.n_agents,
                        :self.n_agents] = obs_mask[:, :self.n_agents]
        obs_mask_padded[:self.n_agents,
                        self.max_n_agents:self.max_n_agents + self.n_enemies] = (
                            obs_mask[:, self.n_agents:]
        )
        entity_mask = np.ones(self.max_n_agents + self.max_n_enemies,
                              dtype=np.uint8)
        entity_mask[:self.n_agents] = 0
        entity_mask[self.max_n_agents:self.max_n_agents + self.n_enemies] = 0
        return obs_mask_padded, entity_mask

    def get_entities(self):
        """
        Returns list of agent entities and enemy entities in the map (all entities are a fixed size)
        All entities together form the global state
        For decentralized execution agents should only have access to the
        entities specified by get_masks()
        """
        all_units = list(self.agents.items()) + list(self.enemies.items())

        nf_entity = self.get_entity_size()

        center_x = self.map_x / 2
        center_y = self.map_y / 2
        com_x = sum(unit.pos.x for u_i, unit in all_units) / len(all_units)
        com_y = sum(unit.pos.y for u_i, unit in all_units) / len(all_units)
        max_dist_com = max(self.distance(unit.pos.x, unit.pos.y, com_x, com_y)
                           for u_i, unit in all_units)

        entities = []
        avail_actions = self.get_avail_actions()
        for u_i, unit in all_units:
            entity = np.zeros(nf_entity, dtype=np.float32)
            # entity tag
            if u_i < self.n_agents:
                tag = self.ally_tags[u_i]
            else:
                tag = self.enemy_tags[u_i - self.n_agents]
            entity[tag] = 1
            ind = self.max_n_agents + self.max_n_enemies
            # available actions (if user controlled entity)
            if u_i < self.n_agents:
                for ac_i in range(self.n_actions - 2):
                    entity[ind + ac_i] = avail_actions[u_i][2 + ac_i]
            ind += self.n_actions - 2
            # unit type
            if self.unit_type_bits > 0:
                type_id = self.unit_type_ids[unit.unit_type]
                entity[ind + type_id] = 1
                ind += self.unit_type_bits
            if unit.health > 0:  # otherwise dead, return all zeros
                # health and shield
                if self.obs_all_health or self.obs_own_health:
                    entity[ind] = unit.health / unit.health_max
                    if ((self.shield_bits_ally > 0 and u_i < self.n_agents) or
                            (self.shield_bits_enemy > 0 and
                             u_i >= self.n_agents)):
                        entity[ind + 1] = unit.shield / unit.shield_max
                    ind += 1 + int(self.shield_bits_ally or
                                   self.shield_bits_enemy)
                # x-y positions
                entity[ind] = (unit.pos.x - center_x) / self.max_distance_x
                entity[ind + 1] = (unit.pos.y - center_y) / self.max_distance_y
                entity[ind + 2] = (unit.pos.x - com_x) / max_dist_com
                entity[ind + 3] = (unit.pos.y - com_y) / max_dist_com
                ind += 4
                if self.obs_pathing_grid:
                    entity[
                        ind:ind + self.n_obs_pathing
                    ] = self.get_surrounding_pathing(unit)
                    ind += self.n_obs_pathing
                if self.obs_terrain_height:
                    entity[ind:] = self.get_surrounding_height(unit)

            entities.append(entity)
            # pad entities to fixed number across episodes (for easier batch processing)
            if u_i == self.n_agents - 1:
                entities += [np.zeros(nf_entity, dtype=np.float32)
                             for _ in range(self.max_n_agents -
                                            self.n_agents)]
            elif u_i == self.n_agents + self.n_enemies - 1:
                entities += [np.zeros(nf_entity, dtype=np.float32)
                             for _ in range(self.max_n_enemies -
                                            self.n_enemies)]

        return entities

    def get_entity_size(self):
        nf_entity = self.max_n_agents + self.max_n_enemies  # tag
        nf_entity += self.n_actions - 2  # available actions minus those that are always available
        nf_entity += self.unit_type_bits  # unit type
        # below are only observed for alive units (else zeros)
        if self.obs_all_health or self.obs_own_health:
            nf_entity += 1 + int(self.shield_bits_ally or self.shield_bits_enemy)  # health and shield
        nf_entity += 4  # global x-y coords + rel x-y to center of mass of all agents (normalized by furthest agent's distance)
        if self.obs_pathing_grid:
            nf_entity += self.n_obs_pathing  # local pathing
        if self.obs_terrain_height:
            nf_entity += self.n_obs_height  # local terrain
        return nf_entity

    def get_obs_agent(self, agent_id, global_obs=False):

        unit = self.get_unit_by_id(agent_id)

        move_feats = np.zeros(self.move_feats_len, dtype=np.float32)  # exclude no-op & stop
        enemy_feats = np.zeros((self.n_enemies, self.nf_en), dtype=np.float32)
        ally_feats = np.zeros((self.n_agents - 1, self.nf_al), dtype=np.float32)
        own_feats = np.zeros(self.nf_own, dtype=np.float32)

        if unit.health > 0:
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[ind: ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if dist < sight_range and e_unit.health > 0:  # visible and alive

                    if self.obs_id_encoding == "metamix":

                        self.obs_set(category="id", val=e_id, obs=enemy_feats[e_id], obs_type="enemy")
                        if dist < sight_range:
                            self.obs_set(val=avail_actions[self.n_actions_no_attack + e_id],
                                         obs=enemy_feats[e_id],
                                         obs_type="enemy",
                                         category="visible")

                    # Sight range > shoot range
                    enemy_feats[e_id] = self.obs_set(val=avail_actions[self.n_actions_no_attack + e_id],
                                                     obs=enemy_feats[e_id],
                                                     obs_type="enemy",
                                                     category="attackable")

                    self.obs_set(category="distance_sc2", val=dist / sight_range, obs=enemy_feats[e_id],
                                 obs_type="enemy")
                    self.obs_set(category="x_sc2", val=(e_x - x) / sight_range, obs=enemy_feats[e_id], obs_type="enemy")
                    self.obs_set(category="y_sc2", val=(e_y - y) / sight_range, obs=enemy_feats[e_id], obs_type="enemy")

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="unit_global",
                                     val=self.get_unit_type_id_metamix(e_unit, False, self.min_unit_type) + 1,
                                     obs=enemy_feats[e_id], obs_type="enemy")

                    if self.obs_all_health:
                        self.obs_set(category="health", val=e_unit.health / e_unit.health_max, obs=enemy_feats[e_id],
                                     obs_type="enemy")

                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            self.obs_set(category="shield", val=e_unit.shield / max_shield, obs=enemy_feats[e_id],
                                         obs_type="enemy")

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        self.obs_set(category="unit_local", val=np.eye(self.unit_type_bits)[type_id],
                                     obs=enemy_feats[e_id], obs_type="enemy")

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)

                if dist < sight_range and al_unit.health > 0:  # visible and alive

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="id", val=al_id, obs=ally_feats[i],
                                     obs_type="ally")  # NEW: have to do this for grid-based input!

                    self.obs_set(category="visible", val=1, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="distance_sc2", val=dist / sight_range, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="x_sc2", val=(al_x - x) / sight_range, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="y_sc2", val=(al_y - y) / sight_range, obs=ally_feats[i], obs_type="ally")

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="unit_global",
                                     val=self.get_unit_type_id_metamix(al_unit, True, self.min_unit_type) + 1,
                                     obs=ally_feats[i], obs_type="ally")

                    if self.obs_all_health:
                        self.obs_set(category="health", val=al_unit.health / al_unit.health_max, obs=ally_feats[i],
                                     obs_type="ally")

                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            self.obs_set(category="shield", val=al_unit.shield / max_shield, obs=ally_feats[i],
                                         obs_type="ally")

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        self.obs_set(category="unit_local", val=np.eye(self.unit_type_bits)[type_id], obs=ally_feats[i],
                                     obs_type="ally")

                    if self.obs_last_action:
                        self.obs_set(category="last_action", val=self.last_action[al_id], obs=ally_feats[i],
                                     obs_type="ally")

        # if unit.health > 0: # otherwise dead, return all zeros
        #     x = unit.pos.x
        #     y = unit.pos.y
        #     sight_range = self.unit_sight_range(agent_id)
        #
        #     # Movement features
        #     avail_actions = self.get_avail_agent_actions(agent_id)
        #     for m in range(self.n_actions_move):
        #         move_feats[m] = avail_actions[m + 2]
        #
        #     ind = self.n_actions_move
        #
        #     if self.obs_pathing_grid:
        #         move_feats[ind: ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
        #         ind += self.n_obs_pathing
        #
        #     if self.obs_terrain_height:
        #         move_feats[ind:] = self.get_surrounding_height(unit)
        #
        #     # Enemy features
        #     for e_id, e_unit in self.enemies.items():
        #         e_x = e_unit.pos.x
        #         e_y = e_unit.pos.y
        #         dist = self.distance(x, y, e_x, e_y)
        #
        #         if dist < sight_range and e_unit.health > 0: # visible and alive
        #             # Sight range > shoot range
        #             enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id] # available
        #             enemy_feats[e_id, 1] = dist / sight_range # distance
        #             enemy_feats[e_id, 2] = (e_x - x) / sight_range # relative X
        #             enemy_feats[e_id, 3] = (e_y - y) / sight_range # relative Y
        #             if self.obs_id_encoding == "metamix":
        #                 enemy_feats[e_id, 4] = self.get_unit_type_id_metamix(e_unit, False, self.min_unit_type) + 1
        #                 ind = 5
        #             else:
        #                 ind = 4
        #             if self.obs_all_health:
        #                 enemy_feats[e_id, ind] = e_unit.health / e_unit.health_max # health
        #                 ind += 1
        #                 if self.shield_bits_enemy > 0:
        #                     max_shield = self.unit_max_shield(e_unit)
        #                     enemy_feats[e_id, ind] = e_unit.shield / max_shield # shield
        #                     ind += 1
        #
        #             if self.unit_type_bits > 0:
        #                 type_id = self.get_unit_type_id(e_unit, False)
        #                 enemy_feats[e_id, ind + type_id] = 1 # unit type
        #
        #     # Ally features
        #     al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        #     for i, al_id in enumerate(al_ids):
        #
        #         al_unit = self.get_unit_by_id(al_id)
        #         al_x = al_unit.pos.x
        #         al_y = al_unit.pos.y
        #         dist = self.distance(x, y, al_x, al_y)
        #
        #         if dist < sight_range and al_unit.health > 0: # visible and alive
        #
        #             ally_feats[i, 0] = 1 # visible
        #             ally_feats[i, 1] = dist / sight_range # distance
        #             ally_feats[i, 2] = (al_x - x) / sight_range # relative X
        #             ally_feats[i, 3] = (al_y - y) / sight_range # relative Y
        #
        #             if self.obs_id_encoding == "metamix":
        #                 ally_feats[i, 4] = self.get_unit_type_id_metamix(al_unit, True, self.min_unit_type) + 1
        #                 ind = 5
        #             else:
        #                 ind = 4
        #
        #             if self.obs_all_health:
        #                 ally_feats[i, ind] = al_unit.health / al_unit.health_max # health
        #                 ind += 1
        #                 if self.shield_bits_ally > 0:
        #                     max_shield = self.unit_max_shield(al_unit)
        #                     ally_feats[i, ind] = al_unit.shield / max_shield # shield
        #                     ind += 1
        #
        #             if self.unit_type_bits > 0:
        #                 type_id = self.get_unit_type_id(al_unit, True)
        #                 ally_feats[i, ind + type_id] = 1
        #                 ind += self.unit_type_bits
        #
        #             if self.obs_last_action:
        #                 ally_feats[i, ind:] = self.last_action[al_id]
        #
        #     # Own features
        #     ind = 0
        #     if self.obs_own_health:
        #         own_feats[ind] = unit.health / unit.health_max
        #         ind += 1
        #         if self.shield_bits_ally > 0:
        #             max_shield = self.unit_max_shield(unit)
        #             own_feats[ind] = unit.shield / max_shield
        #             ind += 1
        #
        #     if self.obs_id_encoding == "metamix":
        #         own_feats[ind] = self.get_unit_type_id_metamix(unit, True, self.min_unit_type) + 1
        #         ind += 1
        #
        #     if self.unit_type_bits > 0:
        #         type_id = self.get_unit_type_id(unit, True)
        #         own_feats[ind + type_id] = 1

        if self.debug_inputs:
            print("***************************************")
            print("Agent: ", agent_id)
            print("Available Actions\n", self.get_avail_agent_actions(agent_id))
            print("Move feats\n", move_feats)
            print("Enemy feats\n", enemy_feats)
            print("Ally feats\n", ally_feats)
            print("Own feats\n", own_feats)
            print("***************************************")

        # Optional: id channels, terrain channel, multiple occupancy, allied last action channel

        if self.obs_decoder is not None and self.obs_decoder.split("_")[0] == "grid":
            self.rasterise_grid(ally_feats,
                                enemy_feats)
            self.resolve_multiple_occupancy(ally_feats,
                                            enemy_feats)
        else:
            if self.obs_grid_rasterise or self.obs_grid_rasterise_debug:
                self.rasterise_grid(ally_feats,
                                    enemy_feats)

            if self.obs_resolve_multiple_occupancy or self.obs_resolve_multiple_occupancy_debug:
                self.resolve_multiple_occupancy(ally_feats,
                                                enemy_feats)

        if not self.obs_decode_on_the_fly:
            assert False, "Not currently implemented!"

        else:
            agent_obs = np.concatenate((ally_feats.flatten(),
                                        enemy_feats.flatten(),
                                        move_feats.flatten(),
                                        own_feats.flatten(),
                                        ))
            return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            Terran.Marine: 15,
            Terran.Marauder: 25,
            Terran.Medivac: 200,  # max energy
            Protoss.Stalker: 35,
            Protoss.Zealot: 22,
            Protoss.Colossus: 24,
            Zerg.Hydralisk: 10,
            Zerg.Zergling: 11,
            Zerg.Baneling: 1
        }
        return switcher.get(unit.unit_type, 15)

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat

        nf_al = 5 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                if al_unit.energy_max > 0.0:
                    ally_state[al_id, 1] = al_unit.energy / al_unit.energy_max
                ally_state[al_id, 2] = al_unit.weapon_cooldown / self.unit_max_cooldown(al_unit)
                ally_state[al_id, 3] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 4] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 5
                if self.shield_bits_ally > 0:
                    ally_state[al_id, ind] = (
                        al_unit.shield / al_unit.shield_max
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.unit_type_ids[al_unit.unit_type]
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    enemy_state[e_id, ind] = (
                        e_unit.shield / e_unit.shield_max
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.unit_type_ids[e_unit.unit_type]
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)

        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state

    def get_obs_size(self):
        """Returns the size of the observation."""
        nf_al = 4 + self.unit_type_bits
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally
            nf_en += 1 + self.shield_bits_enemy

        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1

        if self.obs_last_action:
            nf_al += self.n_actions

        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        enemy_feats = self.n_enemies * nf_en
        ally_feats = (self.n_agents - 1) * nf_al

        return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 5 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            if unit.unit_type == Terran.Medivac:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if not t_unit.is_flying
                ]
                dist_offset = 0
            else:
                target_items = list(self.enemies.items())
                dist_offset = self.n_agents

            for t_id, t_unit in target_items:
                dist = self.dist_mtx[agent_id, t_id + dist_offset]
                if dist <= shoot_range:
                    if unit.unit_type == Terran.Medivac:
                        tag = self.ally_tags[t_id]
                    else:
                        tag = self.enemy_tags[t_id]
                    avail_actions[tag + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.max_n_agents):
            if agent_id < self.n_agents:
                avail_agent = self.get_avail_agent_actions(agent_id)
            else:
                avail_agent = [1] + [0] * (self.n_actions - 1)
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def _kill_all_units(self):
        """Kill all units on the map."""
        self._obs = self._controller.observe()
        tags = [u.tag for u in self._obs.observation.raw_data.units]
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=tags))
        ]
        self.try_controller_step(lambda: self._controller.debug(debug_command))

        while(len(self._obs.observation.raw_data.units) != 0):
            # don't care about step success bc either way map is free of agents
            # (kill command was successful or full restart)
            self.try_controller_step(n_steps=2)

    def init_units(self):
        """Initialise the units."""
        while True:
            self._kill_all_units()

            cmds = []
            # initialize largest possible army to count maximum possible rewards
            ally_army, enemy_army = self.gen_armies_fn(
                max_types_and_units=not self.max_reward_init)
            self.n_agents = 0
            for num, unit_type, pos in ally_army:
                sc_pos = sc_common.Point2D(x=self.map_center[0] + pos[0],
                                           y=self.map_center[1] + pos[1])
                unit_type_id = get_unit_type_by_name(unit_type)
                cmd = d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=unit_type_id,
                        owner=1,
                        pos=sc_pos,
                        quantity=num))
                cmds.append(cmd)
                self.n_agents += num
            self.n_enemies = 0
            for num, unit_type, pos in enemy_army:
                sc_pos = sc_common.Point2D(x=self.map_center[0] + pos[0],
                                           y=self.map_center[1] + pos[1])
                unit_type_id = get_unit_type_by_name(unit_type)
                cmd = d_pb.DebugCommand(
                    create_unit=d_pb.DebugCreateUnit(
                        unit_type=unit_type_id,
                        owner=2,
                        pos=sc_pos,
                        quantity=num))
                cmds.append(cmd)
                self.n_enemies += num
            self._controller.debug(cmds)
            while(len(self._obs.observation.raw_data.units) !=
                  self.n_agents + self.n_enemies):
                step_success = self.try_controller_step(n_steps=1)
                if not step_success:
                    # StarCraft crashed so we need to retry initialization
                    # rather than wait here indefinitely
                    break

            if not step_success:
                continue
            if self.max_reward_init:
                break
            else:
                for unit in self._obs.observation.raw_data.units:
                    if unit.owner == 2:
                        self.max_reward += unit.health_max + unit.shield_max
            self.max_reward_init = True

        self.agents = {}
        self.enemies = {}

        if self.entity_scheme and self.random_tags:
            # assign random tags to agents (used for identifying entities as well as targets for actions)
            self.enemy_tags = np.random.choice(np.arange(self.max_n_enemies),
                                               size=self.n_enemies,
                                               replace=False)
            self.ally_tags = np.random.choice(np.arange(self.max_n_enemies,
                                                        self.max_n_enemies + self.max_n_agents),
                                              size=self.n_agents,
                                              replace=False)
        else:
            self.enemy_tags = np.arange(self.n_enemies)
            self.ally_tags = np.arange(self.max_n_enemies,
                                       self.max_n_enemies + self.n_agents)
        ally_units = [
            unit
            for unit in self._obs.observation.raw_data.units
            if unit.owner == 1
        ]
        ally_units_sorted = sorted(
            ally_units,
            key=attrgetter("unit_type", "pos.x", "pos.y"),
            reverse=False,
        )

        for i in range(len(ally_units_sorted)):
            self.agents[i] = ally_units_sorted[i]
            if self.debug:
                logging.debug(
                    "Unit {} is {}, x = {}, y = {}".format(
                        len(self.agents),
                        self.agents[i].unit_type,
                        self.agents[i].pos.x,
                        self.agents[i].pos.y,
                    )
                )

        for unit in self._obs.observation.raw_data.units:
            if unit.owner == 2:
                self.enemies[len(self.enemies)] = unit

    # def _init_enemy_strategy(self, ally_spawn_center):
    #     tags = [u.tag for u in self.enemies.values()]

    #     cmd = r_pb.ActionRawUnitCommand(
    #         ability_id=actions["attack"],
    #         target_world_space_pos=ally_spawn_center,
    #         unit_tags=tags,
    #         queue_command=False)
    #     sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
    #     req_actions = sc_pb.RequestAction(actions=[sc_action])
    #     self.try_controller_step(lambda: self._controller.actions(req_actions))

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (n_ally_alive == 0 and n_enemy_alive > 0 or
                self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0 or
                self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if Terran.Medivac not in self.unit_type_ids:
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != Terran.Medivac)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != Terran.Medivac)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == Terran.Medivac:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        import dill
        from functools import partial

        obs_decoder = None
        if self.obs_decoder is None:
            pass
        elif self.obs_decoder.split("_")[0] == "grid":
            decompressor = partial(self._grid_obs_metamix,
                                   scenario="_".join(self.obs_decoder.split("_")[1:]),
                                   create_channels=self.create_channels)
            # find number of channels by forwarding dummy input
            n_channels, grid_width, grid_height = decompressor(th.zeros((1, 1, self.get_obs_size())),
                                                               get_size_only=True).shape[-3:]
            obs_decoder = dill.dumps(partial(self.grid_obs_decoder,
                                             width=self.obs_grid_shape[0],
                                             height=self.obs_grid_shape[1],
                                             n_channels=n_channels,
                                             ally_feat_size=self.ally_feats_size,
                                             enemy_feat_size=self.enemy_feats_size,
                                             move_feat_size=self.move_feats_size,
                                             own_feat_size=self.own_feats_size,
                                             compressed=self.obs_decode_on_the_fly,
                                             decompressor=decompressor,
                                             obs_explode=self.obs_explode,
                                             obs_get=self.obs_get))
        elif self.obs_decoder == "transformer_metamix1":
            obs_decoder = dill.dumps(partial(self.transformer_obs_decoder,
                                             move_feat_size=self.move_feats_size,
                                             own_feat_size=self.own_feats_size,
                                             n_agents=self.n_agents,
                                             nf_al=self.nf_al,
                                             n_enemies=self.n_enemies,
                                             nf_en=self.nf_en,
                                             ))

        actions_decoder_grid = dill.dumps(partial(self.actions_decoder_grid,
                                                  obs_grid_shape=self.obs_grid_shape,
                                                  n_actions_no_attack=self.n_actions_no_attack,
                                                  obs_get=self.obs_get,
                                                  obs_explode=self.obs_explode,
                                                  mask_val=float("-999999")
                                                  ))

        actions_encoder_grid = dill.dumps(partial(self.actions_encoder_grid,
                                                  obs_grid_shape=self.obs_grid_shape,
                                                  n_actions_no_attack=self.n_actions_no_attack,
                                                  obs_get=self.obs_get,
                                                  obs_explode=self.obs_explode
                                                  ))

        avail_actions_encoder_grid = dill.dumps(partial(self.avail_actions_encoder_grid,
                                                        obs_grid_shape=self.obs_grid_shape,
                                                        n_actions_no_attack=self.n_actions_no_attack,
                                                        obs_get=self.obs_get,
                                                        obs_explode=self.obs_explode,
                                                        mask_val=float("-999999")
                                                        ))

        env_info = {"state_shape": self.get_state_size(),
                    "state_decoder": dill.dumps(self.state_decoder),
                    "obs_shape": self.get_obs_size(),
                    "obs_shape_decoded": self.get_obs_size() if self.obs_decoder is None else (
                        n_channels, grid_width, grid_height),
                    "obs_decoder": obs_decoder,
                    # "actions_encoder": actions_encoder,
                    "actions_decoder_grid": actions_decoder_grid,
                    "actions_encoder_grid": actions_encoder_grid,
                    "avail_actions_encoder_grid": avail_actions_encoder_grid,
                    # "avail_actions_decoder_flat": avail_actions_decoder_flat,
                    "n_actions": self.n_actions_encoding,  # self.get_total_actions(),
                    "n_actions_no_attack": self.n_actions_no_attack,
                    "n_available_actions": self.n_available_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    # @staticmethod
    # def _grid_rasterise_debug(ally_feats, enemy_feats, obs_grid_shape, obs_get, obs_set, to_grid_coords, from_grid_coords):
    #     # debug function: use to overwrite real coordinates with grid-resolution coordinates!
    #     width, height = obs_grid_shape
    #
    #     for al_id, al_feat in enumerate(ally_feats):
    #         if obs_get(category="visible", obs=al_feat, obs_type="ally") == 1.0:
    #             x_sc2 = obs_get(category="x_sc2", obs=al_feat, obs_type="ally")
    #             y_sc2 = obs_get(category="y_sc2", obs=al_feat, obs_type="ally")
    #             x, y = to_grid_coords(x_sc2, y_sc2)
    #             raster_x, raster_y = from_grid_coords(x, y)
    #             obs_set(category="distance_sc2", val=(raster_x**2 + raster_y**2)**0.5, obs=al_feat, obs_type="ally")
    #             obs_set(category="x_sc2", val=raster_x, obs=al_feat, obs_type="ally")
    #             obs_set(category="y_sc2", val=raster_y, obs=al_feat, obs_type="ally")
    #
    #     for en_id, en_feat in enumerate(enemy_feats):
    #         if obs_get(category="visible", obs=en_feat, obs_type="enemy") == 1.0:
    #             x_sc2 = obs_get(category="x_sc2", obs=en_feat, obs_type="enemy")
    #             y_sc2 = obs_get(category="y_sc2", obs=en_feat, obs_type="enemy")
    #             x, y = to_grid_coords(x_sc2, y_sc2)
    #             raster_x, raster_y = from_grid_coords(x, y)
    #             obs_set(category="distance_sc2", val=(raster_x**2 + raster_y**2)**0.5, obs=en_feat, obs_type="enemy")
    #             obs_set(category="x_sc2", val=raster_x, obs=en_feat, obs_type="enemy")
    #             obs_set(category="y_sc2", val=raster_y, obs=en_feat, obs_type="enemy")
    #
    #     return ally_feats, enemy_feats

    @staticmethod
    def _multiple_occupancy(ally_feats, enemy_feats, obs_grid_shape, obs_get, obs_set, to_grid_coords, from_grid_coords,
                            debug):  # deal with multiple occupancy
        """
        Resolve multiple occupancy trouble from grid rep

        :param enemy_feats:
        :return:
        """
        print("X")
        from collections import defaultdict
        pos_hash = defaultdict(lambda: [])
        width, height = obs_grid_shape

        multiples = set([])
        for al_id, al_feat in enumerate(ally_feats):
            if obs_get(category="visible", obs=al_feat, obs_type="ally") == 1.0:  # ally visible at all
                x_sc2 = obs_get(category="x_sc2", obs=al_feat, obs_type="ally")
                y_sc2 = obs_get(category="y_sc2", obs=al_feat, obs_type="ally")
                x, y = to_grid_coords(x_sc2, y_sc2)
                if len(pos_hash[(x, y)]) > 0:
                    multiples.add((x, y))
                pos_hash[(x, y)].append(dict(utype="ally", id=al_id, feat=al_feat))

        for en_id, en_feat in enumerate(enemy_feats):
            if obs_get(category="visible", obs=en_feat, obs_type="enemy") == 1.0:  # ally visible at all
                x_sc2 = obs_get(category="x_sc2", obs=en_feat, obs_type="enemy")
                y_sc2 = obs_get(category="y_sc2", obs=en_feat, obs_type="enemy")
                x, y = to_grid_coords(x_sc2, y_sc2)
                if len(pos_hash[(x, y)]) > 0:
                    multiples.add((x, y))
                pos_hash[(x, y)].append(dict(utype="enemy", id=en_id, feat=en_feat))

        def _gen_search_coords(_coord, width, height):
            """
            iterate through neighbouring coords, at increasing radius
            """
            for r in range(1, max(height, width)):
                angle_res = 4 * (r + 1)
                for ar in range(0, angle_res):
                    theta = (2 * math.pi / angle_res) * ar
                    pos = _coord[0] + 0.5 + r * math.sin(theta), _coord[1] + 0.5 + r * math.cos(theta)
                    if (int(pos[0]) >= 0 and int(pos[1]) >= 0) and (int(pos[0]) < width and int(pos[1]) < height):
                        yield pos
            pass

        # import numpy as np
        # dbg_grid = np.zeros((width, height))
        # for k, v in pos_hash.items():
        #    dbg_grid[int(k[0]), int(k[1])] = len(v)
        # print(dbg_grid)

        def _pref_sort(lst, obs_id_encoding="metamix"):
            assert obs_id_encoding == "metamix", "not implemented!"
            # arrange elements such that the one least critical to be relocated is popped first
            allegiance_dict = {"ally": 0, "enemy": 1}
            type_dict = {1: 0, 2: 0, 3: 1}  # prioritise long range units to be relocated (1: short range unit)
            try:
                tmp = sorted(lst, key=lambda row: type_dict[
                    int(obs_get(category="unit_global", obs=row["feat"], obs_type="enemy"))], reverse=True)
            except Exception as e:
                a = 4
                pass
            tmp = sorted(tmp, key=lambda row: allegiance_dict[row["utype"]], reverse=True)
            return tmp

        # now deal with multiple occupancy
        for multiple in multiples:
            while len(pos_hash[multiple]) > 1:
                pos_hash[multiple] = _pref_sort(pos_hash[multiple])
                unit = pos_hash[multiple].pop()
                for coord in _gen_search_coords(multiple, height, width):
                    if (int(coord[0]), int(coord[1])) not in pos_hash:
                        pos_hash[(int(coord[0]), int(coord[1]))].append(unit)
                        if debug:
                            x, y = int(coord[0]), int(coord[1])
                            dist = (x ** 2 + y ** 2) ** 0.5
                            feats = ally_feats if unit["utype"] == "ally" else enemy_feats
                            obs_set(category="distance_sc2", val=dist, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="x_sc2", val=x, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="y_sc2", val=y, obs=feats[unit["id"]], obs_type=unit["utype"])
                        else:
                            print("x")
                            x, y = int(coord[0]), int(coord[1])
                            dist = (x ** 2 + y ** 2) ** 0.5
                            x_raster, y_raster = from_grid_coords(coord[0],
                                                                  coord[1])
                            dist_raster = (x_raster ** 2 + y_raster ** 2) ** 0.5
                            feats = ally_feats if unit["utype"] == "ally" else enemy_feats
                            obs_set(category="distance_sc2_raster", val=dist_raster, obs=feats[unit["id"]],
                                    obs_type=unit["utype"])
                            obs_set(category="x_sc2_raster", val=x_raster, obs=feats[unit["id"]],
                                    obs_type=unit["utype"])
                            obs_set(category="y_sc2_raster", val=y_raster, obs=feats[unit["id"]],
                                    obs_type=unit["utype"])
                            obs_set(category="distance_grid", val=dist, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="x_grid", val=x, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="y_grid", val=y, obs=feats[unit["id"]], obs_type=unit["utype"])
                        break

        # import numpy as np
        # dbg_grid = np.zeros((width, height))
        # for k, v in pos_hash.items():
        #    dbg_grid[int(k[0]), int(k[1])] = len(v)
        # print(dbg_grid)

        return ally_feats, enemy_feats

    @staticmethod
    def _grid_obs_orig(ally_feats,
                       enemy_feats,
                       width,
                       height,
                       n_agents,
                       n_enemies,
                       obs_all_health,
                       device=None):
        n_channels = 5

        ally_feats_batch = ally_feats.contiguous().view(-1, n_agents - 1, ally_feats.shape[-1] // (n_agents - 1))
        enemy_feats_batch = enemy_feats.contiguous().view(-1, n_enemies, enemy_feats.shape[-1] // n_enemies)

        channels = ally_feats_batch.new(ally_feats_batch.shape[0], n_channels, width, height).zero_()

        for al_id in range(ally_feats_batch.shape[1]):
            x, y = SC2._to_grid_coords(ally_feats_batch[:, al_id, 2], ally_feats_batch[:, al_id, 3], width, height)
            mask = (ally_feats_batch[:, al_id, 0] == 1).float()  # visibility mask
            channels[:, 0, x, y] = ally_feats_batch[:, al_id, 4] * mask
            if obs_all_health:
                channels[:, 1, x, y] = ally_feats_batch[:, al_id, 5] * mask

        for en_id in range(enemy_feats_batch.shape[1]):
            x, y = SC2._to_grid_coords(enemy_feats_batch[:, en_id, 2], enemy_feats_batch[:, en_id, 3], width, height)
            mask = (enemy_feats_batch[:, en_id].sum(dim=-1) != 0).float()  # visibility mask
            channels[:, 2, x, y] = enemy_feats_batch[:, en_id, 4] * mask
            channels[:, 3, x, y] = enemy_feats_batch[:, en_id, 0] * mask + 1.0
            if obs_all_health:
                channels[:, 4, x, y] = enemy_feats_batch[:, en_id, 5] * mask

        return channels.view(*ally_feats.shape[:2], n_channels, width, height)

    @staticmethod
    def _obs_explode(obs, n_allies, n_enemies, nf_al, nf_en, ally_feats_size, enemy_feats_size, move_feats_len):
        # decompose obs into its constituents parts
        ctr = 0
        obs_dict = {}
        obs_dict["allies"] = obs[..., :ally_feats_size].view(*obs.shape[:-1], n_allies, nf_al)
        ctr += ally_feats_size
        obs_dict["enemies"] = obs[..., ctr:ctr + enemy_feats_size].view(*obs.shape[:-1], n_enemies, nf_en)
        ctr += enemy_feats_size
        obs_dict["move"] = obs[..., ctr:ctr + move_feats_len]
        ctr += move_feats_len
        obs_dict["own"] = obs[..., ctr:]
        return obs_dict

    @staticmethod
    def _create_channels__slow(label,
                               obs,
                               obs_grid_shape,
                               n_allies,
                               n_enemies,
                               obs_set,
                               obs_get,
                               to_grid_coords,
                               from_grid_coords,
                               obs_explode):

        obs_dict = obs_explode(obs)

        if label == "unit_types":

            # create multiple channels, one for each of the 10 smac unit types across allies and enemies
            n_unit_types = 10
            channels = obs.new(*obs.shape[:-1], n_unit_types, *obs_grid_shape).zero_()
            channels_flat_grid = channels.view(*channels.shape[:-3], -1)

            for _id in range(obs_dict["allies"].shape[-2]):
                feat = obs_dict["allies"][..., _id, :]
                unit_global = obs_get(category="unit_global", obs=feat, obs_type="ally")
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="ally")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="ally")
                visible = obs_get(category="visible", obs=feat, obs_type="ally")
                grid_flat = (unit_global - 1) * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[
                    0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                unit_global = obs_get(category="unit_global", obs=feat, obs_type="enemy")
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                grid_flat = (unit_global - 1) * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[
                    0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible)

            return channels

        elif label == "ally_or_enemy":

            # create two channels, one for ally and one for enemy
            n_roles = 2
            channels = obs.new(*obs.shape[:-1], n_roles, *obs_grid_shape).zero_()
            channels_flat_grid = channels.view(*channels.shape[:-3], -1)

            for _id in range(obs_dict["allies"].shape[-2]):
                feat = obs_dict["allies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="ally")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="ally")
                visible = obs_get(category="visible", obs=feat, obs_type="ally")
                grid_flat = y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                grid_flat = obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible)
            return channels

        elif label == "id":
            # create two channels, one for ally and one for enemy
            n_roles = 2
            channels = obs.new(*obs.shape[:-1], n_roles, *obs_grid_shape).zero_()
            channels_flat_grid = channels.view(*channels.shape[:-3], -1)

            for _id in range(obs_dict["allies"].shape[-2]):
                feat = obs_dict["allies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="ally")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="ally")
                visible = obs_get(category="visible", obs=feat, obs_type="ally")
                unit_id = obs_get(category="id", obs=feat, obs_type="ally") + 1
                grid_flat = y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible * unit_id)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                unit_id = obs_get(category="id", obs=feat, obs_type="enemy") + 1
                grid_flat = obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible * unit_id)
            return channels

        elif label == "precise_pos":
            # create four channels, two for ally x,y and one for enemy
            n_roles = 4
            channels = obs.new(*obs.shape[:-1], n_roles, *obs_grid_shape).zero_()
            channels_flat_grid = channels.view(*channels.shape[:-3], -1)

            for _id in range(obs_dict["allies"].shape[-2]):
                feat = obs_dict["allies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="ally")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="ally")
                x_sc2 = obs_get(category="x_sc2", obs=feat, obs_type="ally")
                y_sc2 = obs_get(category="y_sc2", obs=feat, obs_type="ally")
                visible = obs_get(category="visible", obs=feat, obs_type="ally")
                grid_flat_x = 0 * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                grid_flat_y = 1 * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat_x.long(), visible * x_sc2)
                channels_flat_grid.scatter_(-1, grid_flat_y.long(), visible * y_sc2)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                x_sc2 = obs_get(category="x_sc2", obs=feat, obs_type="enemy")
                y_sc2 = obs_get(category="y_sc2", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                grid_flat_x = 2 * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                grid_flat_y = 3 * obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat_x.long(), visible * x_sc2)
                channels_flat_grid.scatter_(-1, grid_flat_y.long(), visible * y_sc2)

            return channels

        elif label == "health":
            # create two channels, one for ally and one for enemy
            n_roles = 2
            channels = obs.new(*obs.shape[:-1], n_roles, *obs_grid_shape).zero_()
            channels_flat_grid = channels.view(*channels.shape[:-3], -1)

            for _id in range(obs_dict["allies"].shape[-2]):
                feat = obs_dict["allies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="ally")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="ally")
                visible = obs_get(category="visible", obs=feat, obs_type="ally")
                health = obs_get(category="health", obs=feat, obs_type="ally")
                grid_flat = y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible * health)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                health = obs_get(category="health", obs=feat, obs_type="ally")
                grid_flat = obs_grid_shape[0] * obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible * health)
            return channels

        pass

    @staticmethod
    def _create_channels(label_lst,
                         obs,
                         obs_grid_shape,
                         n_allies,
                         n_enemies,
                         obs_set,
                         obs_get,
                         to_grid_coords,
                         from_grid_coords,
                         obs_explode,
                         get_size_only=False
                         ):

        obs_dict = obs_explode(obs)
        x_grid_ally = obs_get(category="x_grid", obs=obs_dict["allies"], obs_type="ally")
        y_grid_ally = obs_get(category="y_grid", obs=obs_dict["allies"], obs_type="ally")
        x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy")
        y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy")
        visible_ally = obs_get(category="visible", obs=obs_dict["allies"], obs_type="ally")
        visible_enemy = obs_get(category="visible", obs=obs_dict["enemies"], obs_type="enemy")

        # create ally and enemy index masks (i.e. only select positions that correspond to visible entities)
        grid_coords_flat_ally = (y_grid_ally * obs_grid_shape[0] + x_grid_ally).long()
        grid_coords_flat_enemy = (y_grid_enemy * obs_grid_shape[0] + x_grid_enemy).long()

        # start filling the cache
        cache = {}
        cache["x_grid_ally"] = x_grid_ally
        cache["y_grid_ally"] = y_grid_ally
        cache["x_grid_enemy"] = x_grid_enemy
        cache["y_grid_enemy"] = y_grid_enemy
        cache["visible_ally"] = visible_ally
        cache["visible_enemy"] = visible_enemy
        cache["grid_coords_flat_ally"] = grid_coords_flat_ally
        cache["grid_coords_flat_enemy"] = grid_coords_flat_enemy
        cache["grid_coords_flat"] = th.cat([grid_coords_flat_ally, grid_coords_flat_enemy], dim=-2).view(
            *grid_coords_flat_ally.shape[:-2],
            -1)
        cache["x_grid"] = th.cat([cache["x_grid_ally"], cache["x_grid_enemy"]], dim=-2)
        cache["y_grid"] = th.cat([cache["y_grid_ally"], cache["y_grid_enemy"]], dim=-2)
        cache["y_grid"] = th.cat([cache["y_grid_ally"], cache["y_grid_enemy"]], dim=-2)
        cache["visible"] = th.cat([cache["visible_ally"], cache["visible_enemy"]], dim=-2).view(
            *cache["visible_ally"].shape[:-2],
            -1)
        cache["ally_enemy_mask"] = cache["x_grid"].clone().zero_()
        cache["ally_enemy_mask"][..., obs_dict["allies"].shape[-2]:, :] = 1  # enemies are 1 in this mask, allies zero
        cache["ally_enemy_mask"] = cache["ally_enemy_mask"].view(*cache["ally_enemy_mask"].shape[:-2],
                                                                 -1)

        # calculate and assign empty memory for all channels
        dim_channels_dict = {"unit_type": {"uncompressed": 10, "compressed": 10},
                             "ally_or_enemy": {"uncompressed": 2, "compressed": 1},
                             "id": {"uncompressed": 2, "compressed": 1},
                             "precise_pos": {"uncompressed": 4, "compressed": 2},
                             "health": {"uncompressed": 2, "compressed": 1},
                             "last_action": {"uncompressed": 2, "compressed": 2},
                             "attackable": {"uncompressed": 1, "compressed": 1}}

        n_channels = sum([dim_channels_dict[label["type"]][label.get("mode", "uncompressed")] for label in label_lst])

        channels_slice_dict = {}
        ctr = 0
        for label in label_lst:
            channels_slice_dict[label["type"]] = slice(ctr, ctr + dim_channels_dict[label["type"]][
                label.get("mode", "uncompressed")])
            ctr += dim_channels_dict[label["type"]][label.get("mode", "uncompressed")]

        channels = obs.new(*obs.shape[:-1], n_channels, *obs_grid_shape).zero_()
        channels_flat_grid = channels.view(*channels.shape[:-3], -1)

        if get_size_only:
            return channels

        for label in label_lst:

            label_type = label["type"]
            label_mode = label.get("mode", "uncompressed")
            label_slice = channels_slice_dict[label_type]

            if label_type == "unit_type":
                # label mode does not matter

                if not ("unit_global" in cache):
                    if not ("unit_global_allies" in cache):
                        cache["unit_global_allies"] = obs_get(category="unit_global", obs=obs_dict["allies"],
                                                              obs_type="ally").long()
                    unit_global_allies = cache["unit_global_allies"]

                    if not ("unit_global_enemies" in cache):
                        cache["unit_global_enemies"] = obs_get(category="unit_global", obs=obs_dict["enemies"],
                                                               obs_type="enemy").long()
                    unit_global_enemies = cache["unit_global_enemies"]

                    cache["unit_global"] = th.cat([unit_global_allies, unit_global_enemies], dim=-2)
                    cache["unit_global"] = cache["unit_global"].view(*cache["unit_global"].shape[:-2],
                                                                     -1)

                unit_global = cache["unit_global"]
                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]

                # TODO: use diverse label types for units!
                gcf = (label_slice.start + unit_global) * obs_grid_shape[0] * obs_grid_shape[1] + grid_coords_flat
                channels_flat_grid.scatter_(-1, gcf, visible)

            elif label_type == "ally_or_enemy":
                visible = cache["visible"]
                grid_coords_flat = cache["grid_coords_flat"]
                ally_enemy_mask = cache["ally_enemy_mask"]

                if label_mode == "compressed":
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * (ally_enemy_mask * 2 - 1))
                elif label_mode == "uncompressed":
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[
                        1]
                    channels_flat_grid.scatter_(-1, gcf, visible)

            elif label_type == "id":

                if not ("unit_id" in cache):
                    if not ("unit_id_allies" in cache):
                        cache["unit_id_allies"] = obs_get(category="id", obs=obs_dict["allies"], obs_type="ally")
                    unit_global_allies = cache["unit_id_allies"]

                    if not ("unit_id_enemies" in cache):
                        cache["unit_id_enemies"] = obs_get(category="id", obs=obs_dict["enemies"], obs_type="enemy")
                    unit_global_enemies = cache["unit_id_enemies"]

                    cache["unit_id"] = th.cat([unit_global_allies, unit_global_enemies], dim=-2).float()
                    cache["unit_id"] = cache["unit_id"].view(*cache["unit_id"].shape[:-2],
                                                             -1)

                grid_coords_flat = cache["grid_coords_flat"]
                unit_id = cache["unit_id"]

                if label_mode == "compressed":
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * ((1 - ally_enemy_mask) * 2 - 1) * unit_id)

                elif label_mode == "uncompressed":
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[
                        1]
                    channels_flat_grid.scatter_(-1, gcf, visible * unit_id)

            elif label_type == "precise_pos":

                if not ("x_sc2" in cache):
                    cache["x_sc2_ally"] = obs_get(category="x_sc2", obs=obs_dict["allies"], obs_type="ally")
                    cache["x_sc2_enemy"] = obs_get(category="x_sc2", obs=obs_dict["enemies"], obs_type="enemy")
                    cache["x_sc2"] = th.cat([cache["x_sc2_ally"], cache["x_sc2_enemy"]], dim=-2)
                    cache["x_sc2"] = cache["x_sc2"].view(*cache["x_sc2"].shape[:-2],
                                                         -1)

                if not ("y_sc2" in cache):
                    cache["y_sc2_ally"] = obs_get(category="y_sc2", obs=obs_dict["allies"], obs_type="ally")
                    cache["y_sc2_enemy"] = obs_get(category="y_sc2", obs=obs_dict["enemies"], obs_type="enemy")
                    cache["y_sc2"] = th.cat([cache["y_sc2_ally"], cache["y_sc2_enemy"]], dim=-2)
                    cache["y_sc2"] = cache["y_sc2"].view(*cache["y_sc2"].shape[:-2],
                                                         -1)

                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]
                x_sc2 = cache["x_sc2"]
                y_sc2 = cache["y_sc2"]

                if label_mode == "compressed":
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * x_sc2)
                    gcf = grid_coords_flat + 1 * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * y_sc2)

                elif label_mode == "uncompressed":
                    ally_enemy_mask = cache["ally_enemy_mask"]
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[
                        1]
                    channels_flat_grid.scatter_(-1, gcf, visible * x_sc2)
                    gcf = grid_coords_flat + 2 * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * y_sc2)

            elif label_type == "last_action":

                if not ("last_action" in cache):
                    cache["last_action_no_attack"] = obs_get(category="last_action", obs=obs_dict["allies"],
                                                             obs_type="ally")

                    cache["last_action_no_attack"] = cache["last_action_no_attack"].view(
                        *grid_coords_flat_ally.shape[:-2],
                        -1)

                if not ("last_attacked_grid" in cache):
                    cache["last_attacked_grid"] = obs_get(category="health", obs=obs_dict["allies"], obs_type="ally")
                    cache["last_attacked_grid"] = cache["last_attacked_grid"].view(*grid_coords_flat_ally.shape[:-2],
                                                                                   -1)

                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]
                last_action_no_attack = cache["last_action_no_attack"]
                last_attacked_grid = cache["last_attacked_grid"]

                if label_mode in ["compressed", "uncompressed"]:
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * last_action_no_attack)
                    channels[..., label_slice.start + 1, :, :] = last_attacked_grid

            elif label_type == "health":

                if not ("health" in cache):
                    cache["health_ally"] = obs_get(category="health", obs=obs_dict["allies"], obs_type="ally")
                    cache["health_enemy"] = obs_get(category="health", obs=obs_dict["enemies"], obs_type="enemy")
                    cache["health"] = th.cat([cache["health_ally"], cache["health_enemy"]], dim=-2)
                    cache["health"] = cache["health"].view(*grid_coords_flat_ally.shape[:-2],
                                                           -1)

                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]
                health = cache["health"]

                if label_mode == "compressed":
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * health)

                elif label_mode == "uncompressed":
                    ally_enemy_mask = cache["ally_enemy_mask"]
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[
                        1]
                    channels_flat_grid.scatter_(-1, gcf, visible * health)

        return channels

    @staticmethod
    def _grid_obs_metamix(obs, create_channels, scenario="metamix1", get_size_only=False):

        if scenario == "metamix1":
            # This encoding has one-hot encoding of agent types (could try binary encoding at a later stage)
            # channels = []
            # channels.append(create_channels(label="unit_types", obs=obs))
            # channels.append(create_channels(label="ally_or_enemy", obs=obs))
            # channels.append(create_channels(label="precise_pos", obs=obs))
            # channels.append(create_channels(label="health", obs=obs))
            # channels.append(create_channels(label="unit_types", obs=obs))
            # output = th.cat(channels, dim=-3)
            return output

        elif scenario == "metamix__3m":

            label_lst = [{"type": "ally_or_enemy", "mode": "compressed"},
                         {"type": "id", "mode": "compressed"},
                         {"type": "health", "mode": "compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__3m_noid":

            label_lst = [{"type": "ally_or_enemy", "mode": "compressed"},
                         {"type": "last_action", "mode": "compressed"},
                         {"type": "multiple_occupancy", "mode": "compressed"},
                         {"type": "health", "mode": "compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__3m_precise":
            label_lst = [{"type": "ally_or_enemy", "mode": "compressed"},
                         {"type": "id", "mode": "compressed"},
                         {"type": "health", "mode": "compressed"},
                         {"type": "precise_pos", "mode": "compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__2s3z":
            label_lst = [{"type": "unit_type", "mode": "compressed"},
                         {"type": "ally_or_enemy", "mode": "compressed"},
                         {"type": "id", "mode": "compressed"},
                         {"type": "health", "mode": "compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            channels_out = th.cat([channels[..., 1:3, :, :],
                                   channels[..., 10:, :, :]], dim=-3)  # remove superfluous type channels
            return channels_out

        elif scenario == "metamix__2s3z_noid_nolastaction":
            label_lst = [{"type": "unit_type", "mode": "compressed"},
                         {"type": "ally_or_enemy", "mode": "compressed"},
                         # {"type": "id", "mode": "compressed"},
                         {"type": "health", "mode": "compressed"}]

            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            channels_out = th.cat([channels[..., 1:3, :, :],
                                   channels[..., 10:, :, :]], dim=-3)  # remove superfluous type channels
            return channels_out


        elif scenario == "metamix__2s3z_noid_nolastaction_nounittype":
            label_lst = [  # {"type": "unit_type", "mode": "compressed"},
                {"type": "ally_or_enemy", "mode": "compressed"},
                # {"type": "id", "mode": "compressed"},
                {"type": "health", "mode": "compressed"}]

            channels_out = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels_out

        elif scenario == "metamix__2s3z_only_ally_or_enemy":
            label_lst = [  # {"type": "unit_type", "mode": "compressed"},
                {"type": "ally_or_enemy", "mode": "compressed"},
                # {"type": "id", "mode": "compressed"},
                # {"type": "health", "mode": "compressed"}
            ]

            channels_out = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels_out

        elif scenario == "metamix__2s3z_empty":
            label_lst = []

            channels_out = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels_out

        elif scenario == "metamix__2s3z_noid":
            label_lst = [{"type": "unit_type", "mode": "compressed"},
                         {"type": "ally_or_enemy", "mode": "compressed"},
                         # {"type":"id", "mode":"compressed"},
                         {"type": "last_action", "mode": "compressed"},
                         # {"type":"multiple_occupancy", "mode":"compressed"},
                         {"type": "health", "mode": "compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            channels_out = th.cat([channels[..., 1:3, :, :],
                                   channels[..., 10:, :, :]], dim=-3)  # remove superfluous type channels
            return channels_out


        elif scenario == "metamix__2s3z_precise":
            # channels = []
            # ally_or_enemy_channels = create_channels(label="ally_or_enemy", obs=obs)
            # ally_or_enemy_channel = ally_or_enemy_channels[...,0:1,:,:] - ally_or_enemy_channels[...,1:2,:,:]
            # channels.append(ally_or_enemy_channel)
            # id_channels = create_channels(label="id", obs=obs)
            # id_channel = id_channels[...,0:1,:,:] - id_channels[...,1:2,:,:]
            # channels.append(id_channel)
            # health_channels = create_channels(label="health", obs=obs)
            # health_channel  = health_channels[...,0:1,:,:] + health_channels[...,1:2,:,:]
            # channels.append(health_channel)
            # precise_pos_channels = create_channels(label="precise_pos", obs=obs)
            # precise_pos_x_channel = precise_pos_channels[...,0:1,:,:] + precise_pos_channels[...,2:3,:,:]
            # precise_pos_y_channel = precise_pos_channels[..., 1:2, :, :] + precise_pos_channels[..., 3:4, :, :]
            # channels.append(precise_pos_x_channel)
            # channels.append(precise_pos_y_channel)
            # output = th.cat(channels, dim=-3)
            return output

    @staticmethod
    def _obs_get(obs, obs_type, category, idxs=None):
        # retrieve values using label
        idx = idxs[obs_type][category]
        if isinstance(idx, tuple):
            ret = obs[..., idx[0]:idx[1]]
        else:
            ret = obs[..., idx:idx + 1]
        return ret[0] if len(ret.shape) == 1 else ret

    @staticmethod
    def _obs_set(obs, val, obs_type, category, idxs=None):
        # retrieve values using label
        idx = idxs[obs_type][category]
        if isinstance(idx, tuple):
            v = obs[..., idx[0]:idx[1]]
            v[:] = val
        else:
            v = obs[..., idx:idx + 1]
            v[:] = val
        return obs

    @staticmethod
    def _to_grid_coords(x, y, obs_grid_shape, x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0)):
        width, height = obs_grid_shape
        x_grid = (x - x_lims[0]) / (x_lims[1] - x_lims[0]) * (width - 1)
        y_grid = (y - y_lims[0]) / (y_lims[1] - y_lims[0]) * (height - 1)
        return x_grid.round(), y_grid.round()

    @staticmethod
    def _from_grid_coords(x_grid, y_grid, obs_grid_shape, x_lims=(-1.0, 1.0), y_lims=(-1.0, 1.0)):
        # assume grid centers
        width, height = obs_grid_shape
        x = x_grid * (x_lims[1] - x_lims[0]) / (width - 1) + x_lims[0]
        y = y_grid * (y_lims[1] - y_lims[0]) / (height - 1) + y_lims[0]
        return x, y

    @staticmethod
    def _grid_rasterise(ally_feats, enemy_feats, obs_grid_shape, obs_get, obs_set, to_grid_coords, from_grid_coords,
                        debug=False):
        # will store
        width, height = obs_grid_shape
        for al_id, al_feat in enumerate(ally_feats):
            if obs_get(category="visible", obs=al_feat, obs_type="ally") == 1.0:
                x_sc2 = obs_get(category="x_sc2", obs=al_feat, obs_type="ally")
                y_sc2 = obs_get(category="y_sc2", obs=al_feat, obs_type="ally")
                x, y = to_grid_coords(x_sc2, y_sc2)
                raster_x, raster_y = from_grid_coords(x, y)
                if not debug:
                    obs_set(category="distance_grid", val=(x ** 2 + y ** 2) ** 0.5, obs=al_feat, obs_type="ally")
                    obs_set(category="x_grid", val=x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_grid", val=y, obs=al_feat, obs_type="ally")
                    obs_set(category="distance_sc2_raster", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=al_feat,
                            obs_type="ally")
                    obs_set(category="x_sc2_raster", val=raster_x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_sc2_raster", val=raster_y, obs=al_feat, obs_type="ally")
                else:
                    obs_set(category="distance_sc2", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=al_feat,
                            obs_type="ally")
                    obs_set(category="x_sc2", val=raster_x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_sc2", val=raster_y, obs=al_feat, obs_type="ally")

        for en_id, en_feat in enumerate(enemy_feats):
            if obs_get(category="visible", obs=en_feat, obs_type="enemy") == 1.0:
                x_sc2 = obs_get(category="x_sc2", obs=en_feat, obs_type="enemy")
                y_sc2 = obs_get(category="y_sc2", obs=en_feat, obs_type="enemy")
                x, y = to_grid_coords(x_sc2, y_sc2)
                raster_x, raster_y = from_grid_coords(x, y)
                if not debug:
                    obs_set(category="distance_grid", val=(x ** 2 + y ** 2) ** 0.5, obs=en_feat, obs_type="enemy")
                    obs_set(category="x_grid", val=x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_grid", val=y, obs=en_feat, obs_type="enemy")
                    obs_set(category="distance_sc2_raster", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=en_feat,
                            obs_type="enemy")
                    obs_set(category="x_sc2_raster", val=raster_x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_sc2_raster", val=raster_y, obs=en_feat, obs_type="enemy")
                else:
                    obs_set(category="distance_sc2", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=en_feat,
                            obs_type="enemy")
                    obs_set(category="x_sc2", val=raster_x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_sc2", val=raster_y, obs=en_feat, obs_type="enemy")

        return ally_feats, enemy_feats