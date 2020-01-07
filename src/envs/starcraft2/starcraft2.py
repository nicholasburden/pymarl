from ..multiagentenv import MultiAgentEnv
from .map_params import get_map_params, map_present
from utils.dict2namedtuple import convert
from operator import attrgetter
from copy import deepcopy
from absl import flags
import numpy as np
import pygame
import sys
import torch as th
import os
import math

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

FLAGS = flags.FLAGS
FLAGS(['main.py'])

_possible_results = {
    sc_pb.Victory: 1,
    sc_pb.Defeat: -1,
    sc_pb.Tie: 0,
    sc_pb.Undecided: 0,
}

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

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

action_move_id = 16     #    target: PointOrUnit
action_attack_id = 23   #    target: PointOrUnit
action_stop_id = 4      #    target: None
action_heal_id = 386    #    target: Unit

'''
StarCraft II
'''

class SC2(MultiAgentEnv):

    def __init__(self, **kwargs):

        args = kwargs
        if isinstance(args, dict):
            args = convert(args)

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
        self.restrict_actions = True #args.restrict_actions

        # For sanity check
        self.debug_inputs = False
        self.debug_rewards = False

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.action_representation = getattr(args, "action_representation", "original")

        # Configuration related to obs featurisation
        self.obs_id_encoding = getattr(args, "obs_id_encoding", "original") # one of: original, metamix
        self.obs_decoder = getattr(args, "obs_decoder", None)  # None: flatten output!
        self.obs_decode_on_the_fly = getattr(args, "obs_decode_on_the_fly", True)
        self.obs_grid_shape = list(map(int, getattr(args, "obs_grid_shape", "1x1").split("x"))) # (width, height)
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
            self.n_actions_encoding = self.obs_grid_shape[0]*self.obs_grid_shape[1] + self.n_actions_no_attack
        elif self.action_representation == "input_grid":
            self.n_actions_encoding = self.obs_grid_shape[0]*self.obs_grid_shape[1] + self.n_actions_no_attack
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
        self._game_info = self.controller.game_info()
        self._map_info = self._game_info.start_raw
        self.map_x = self._map_info.map_size.x
        self.map_y = self._map_info.map_size.y
        self.map_play_area_min = self._map_info.playable_area.p0
        self.map_play_area_max = self._map_info.playable_area.p1
        self.max_distance_x = self.map_play_area_max.x - self.map_play_area_min.x
        self.max_distance_y = self.map_play_area_max.y - self.map_play_area_min.y
        self.terrain_height = np.flip(np.transpose(np.array(list(self._map_info.terrain_height.data)).reshape(self.map_x, self.map_y)), 1)
        self.pathing_grid = np.flip(np.transpose(np.array(list(self._map_info.pathing_grid.data)).reshape(self.map_x, self.map_y)), 1)

        self._episode_count = 0
        self._total_steps = 0

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        self.last_stats = None

        # Calculate feature sizes

        obs_items = {"ally": [], "enemy": [], "own": [] }
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
                dct[name] = (ctr, ctr+length)
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
        self.obs_get = partial(SC2._obs_get,
                               idxs=idxs)

        self.obs_set = partial(SC2._obs_set,
                               idxs=idxs)

        self.to_grid_coords = partial(SC2._to_grid_coords,
                                      obs_grid_shape=self.obs_grid_shape,
                                      )

        self.from_grid_coords = partial(SC2._from_grid_coords,
                                        obs_grid_shape=self.obs_grid_shape,
                                        )

        self.rasterise_grid = partial(SC2._grid_rasterise,
                                      obs_grid_shape=self.obs_grid_shape,
                                      obs_get=self.obs_get,
                                      obs_set=self.obs_set,
                                      to_grid_coords=self.to_grid_coords,
                                      from_grid_coords=self.from_grid_coords,
                                      debug=self.obs_grid_rasterise_debug
                                      )

        self.resolve_multiple_occupancy = partial(SC2._multiple_occupancy,
                                                  obs_grid_shape=self.obs_grid_shape,
                                                  obs_get=self.obs_get,
                                                  obs_set=self.obs_set,
                                                  to_grid_coords=self.to_grid_coords,
                                                  from_grid_coords = self.from_grid_coords,
                                                  debug=self.obs_resolve_multiple_occupancy_debug
                                                  )

        self.obs_explode = partial(SC2._obs_explode,
                                   ally_feats_size=self.ally_feats_size,
                                   enemy_feats_size=self.enemy_feats_size,
                                   move_feats_len=self.move_feats_len,
                                   n_allies=self.n_agents-1,
                                   n_enemies=self.n_enemies,
                                   nf_al=self.nf_al,
                                   nf_en=self.nf_en, )

        self.create_channels = partial(SC2._create_channels,
                                       n_allies=self.n_agents-1,
                                       n_enemies=self.n_enemies,
                                       obs_grid_shape=self.obs_grid_shape,
                                       obs_get=self.obs_get,
                                       obs_set=self.obs_set,
                                       to_grid_coords=self.to_grid_coords,
                                       from_grid_coords=self.from_grid_coords,
                                       obs_explode=self.obs_explode)
        return

    @staticmethod
    def _obs_get(obs, obs_type, category, idxs=None):
        # retrieve values using label
        idx = idxs[obs_type][category]
        if isinstance(idx, tuple):
            ret = obs[..., idx[0]:idx[1]]
        else:
            ret = obs[..., idx:idx+1]
        return ret[0] if len(ret.shape) == 1 else ret

    @staticmethod
    def _obs_set(obs, val, obs_type, category, idxs=None):
        # retrieve values using label
        idx = idxs[obs_type][category]
        if isinstance(idx, tuple):
            v = obs[..., idx[0]:idx[1]]
            v[:] = val
        else:
            v = obs[..., idx:idx+1]
            v[:] = val
        return obs

    def init_ally_unit_types(self, min_unit_type):
        # This should be called once from the init_units function

        self.stalker_id = self.sentry_id = self.zealot_id = self.colossus_id = 0
        self.marine_id = self.marauder_id= self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0

        self.min_unit_type = min_unit_type

        if self.map_type == 'sz' or self.map_type == 's_v_z':
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == 'MMM':
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == 'zealots':
            self.zealot_id = min_unit_type
        elif self.map_type == 'focus_fire':
            self.hydralisk_id = min_unit_type
        elif self.map_type == 'retarget':
            self.stalker_id = min_unit_type
        elif self.map_type == 'colossus':
            self.colossus_id = min_unit_type
        elif self.map_type == 'ze_ba':
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def _launch(self):

        self._run_config = run_configs.get()
        self._map = maps.get(self.map_name)

        # Setting up the interface
        self.interface = sc_pb.InterfaceOptions(
                raw = True, # raw, feature-level data
                score = True)

        self._sc2_proc = self._run_config.start(game_version=self.game_version, window_size=self.window_size)
        self.controller = self._sc2_proc.controller

        # Create the game.
        create = sc_pb.RequestCreateGame(realtime = False,
                random_seed = self.seed,
                local_map=sc_pb.LocalMap(map_path=self._map.path, map_data=self._run_config.map_data(self._map.path)))
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],
                                difficulty=difficulties[self.difficulty])
        self.controller.create_game(create)

        join = sc_pb.RequestJoinGame(race=races[self._agent_race], options=self.interface)
        self.controller.join_game(join)

    def save_replay(self):
        prefix = self.save_replay_prefix or self.map_name
        replay_path = self._run_config.save_replay(self.controller.save_replay(), replay_dir='', prefix=prefix)
        print("Replay saved at: %s" % replay_path)

    def reset(self):
        """Start a new episode."""

        if self.debug_inputs or self.debug_rewards:
            print('------------>> RESET <<------------')

        self._episode_steps = 0
        if self._episode_count > 0:
            # No need to restart for the first episode.
            self._restart()

        self._episode_count += 1

        if self.heuristic:
            self.heuristic_targets = [0] * self.n_agents

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_agent_units = None
        self.previous_enemy_units = None

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        try:
            self._obs = self.controller.observe()
            self.init_units()
        except protocol.ProtocolError:
            self.full_restart()
        except protocol.ConnectionError:
            self.full_restart()

        #print(self.controller.query(q_pb.RequestQuery(abilities=[q_pb.RequestQueryAvailableAbilities(unit_tag=self.agents[0].tag)])))
        #print(self.controller.data_raw())

        return self.get_obs(), self.get_state()

    def _restart(self):

        # Kill and restore all units
        try:
            self.kill_all_units()
            self.controller.step(2)
        except protocol.ProtocolError:
            self.full_restart()
        except protocol.ConnectionError:
            self.full_restart()

    def full_restart(self):
        # End episode and restart a new one
        self._sc2_proc.close()
        self._launch()
        self.force_restarts += 1

    def one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]

    def step(self, actions):

        actions = [int(a) for a in actions]

        self.last_action = self.one_hot(actions, self.n_actions)

        # Collect individual actions
        sc_actions = []
        for a_id, action in enumerate(actions):
            if not self.heuristic:
                agent_action = self.get_agent_action(a_id, action)
            else:
                agent_action = self.get_agent_action_heuristic(a_id, action)
            if agent_action:
              sc_actions.append(agent_action)
        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)

        try:
            res_actions = self.controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self.controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self.controller.observe()
        except protocol.ProtocolError:
            self.full_restart()
            return 0, True, {}
        except protocol.ConnectionError:
            self.full_restart()
            return 0, True, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update what we know about units
        end_game = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        if end_game is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if end_game == 1:
                self.battles_won += 1
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1

            elif end_game == -1:
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self.episode_limit > 0 and self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug_inputs or self.debug_rewards:
            print("Total Reward = %.f \n ---------------------" % reward)

        if self.reward_scale:
            reward /= (self.max_reward / self.reward_scale_rate)

        return reward, terminated, info

    def get_agent_action(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        true_avail_actions = self.get_avail_agent_actions(a_id)
        if true_avail_actions[action] == 0:
            action = 1

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op chosen but the agent's unit is not dead"
            if self.debug_inputs:
                print("Agent %d: Dead"% a_id)
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_stop_id,
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: Stop"% a_id)

        elif action == 2:
            # north
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x, y = y + self._move_amount),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: North"% a_id)

        elif action == 3:
            # south
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x, y = y - self._move_amount),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: South"% a_id)

        elif action == 4:
            # east
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x + self._move_amount, y = y),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: East"% a_id)

        elif action == 5:
            # west
            cmd = r_pb.ActionRawUnitCommand(ability_id = action_move_id,
                    target_world_space_pos = sc_common.Point2D(x = x - self._move_amount, y = y),
                    unit_tags = [tag],
                    queue_command = False)
            if self.debug_inputs:
                print("Agent %d: West"% a_id)
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == 'MMM' and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_id = action_heal_id
            else:
                target_unit = self.enemies[target_id]
                action_id = action_attack_id
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(ability_id = action_id,
                    target_unit_tag = target_tag,
                    unit_tags = [tag],
                    queue_command = False)

            if self.debug_inputs:
                print("Agent %d attacks enemy # %d" % (a_id, target_id))

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_agent_action_heuristic(self, a_id, action):

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag

        target_tag = self.enemies[self.heuristic_targets[a_id]].tag
        action_id = action_attack_id

        cmd = r_pb.ActionRawUnitCommand(ability_id = action_id,
                target_unit_tag = target_tag,
                unit_tags = [tag],
                queue_command = False)

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def reward_battle(self):

        if self.reward_sparse:
            return 0

        #  delta health - delta enemies + delta deaths where value:
        #   if enemy unit dies, add reward_death_value per dead unit
        #   if own unit dies, subtract reward_death_value per dead unit

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        if self.debug_rewards:
            for al_id in range(self.n_agents):
                print("Agent %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_agent_units[al_id].health - self.agents[al_id].health, self.previous_agent_units[al_id].shield - self.agents[al_id].shield))
            print('---------------------')
            for al_id in range(self.n_enemies):
                print("Enemy %d: diff HP = %.f, diff shield = %.f" % (al_id, self.previous_enemy_units[al_id].health - self.enemies[al_id].health, self.previous_enemy_units[al_id].shield - self.enemies[al_id].shield))

        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = self.previous_agent_units[al_id].health + self.previous_agent_units[al_id].shield
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += (prev_health - al_unit.health - al_unit.shield) * neg_scale

        for e_id, e_unit in self.enemies.items():
            if not self.death_tracker_enemy[e_id]:
                prev_health = self.previous_enemy_units[e_id].health + self.previous_enemy_units[e_id].shield
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:

            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths
            reward = abs(reward) # shield regeration
        else:
            if self.debug_rewards:
                print("--------------------------")
                print("Delta enemy: ", delta_enemy)
                print("Delta deaths: ", delta_deaths)
                print("Delta ally: ", - delta_ally)
                print("Reward: ", delta_enemy + delta_deaths)
                print("--------------------------")

            reward = delta_enemy + delta_deaths - delta_ally

        return reward

    def get_total_actions(self):
        return self.n_actions

    def distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def unit_shoot_range(self, agent_id):
        return 6

    def unit_sight_range(self, agent_id):
        return 9

    def unit_max_cooldown(self, agent_id):

        if self.map_type == 'marines':
            return 15

        unit = self.get_unit_by_id(agent_id)
        if unit.unit_type == self.marine_id:
            return 15
        if unit.unit_type == self.marauder_id:
            return 25
        if unit.unit_type == self.medivac_id:
            return 200

        if unit.unit_type == self.stalker_id:
            return 35
        if unit.unit_type == self.zealot_id:
            return 22
        if unit.unit_type == self.colossus_id:
            return 24
        if unit.unit_type == self.sentry_id:
            return 13
        if unit.unit_type == self.hydralisk_id:
            return 10
        if unit.unit_type == self.zergling_id:
            return 11
        if unit.unit_type == self.baneling_id:
            return 1

    def unit_max_shield(self, unit):

        if unit.unit_type == 74 or unit.unit_type == self.stalker_id: # Protoss's Stalker
            return 80
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id: # Protoss's Zaelot
            return 50
        if unit.unit_type == 77 or unit.unit_type == self.sentry_id: # Protoss's Sentry
            return 40
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id: # Protoss's Colossus
            return 150

    def can_move(self, unit, direction):

        m = self._move_amount / 2

        if direction == 0: # north
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == 1: # south
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == 2: # east
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else : # west
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.pathing_grid[x, y] == 0:
            return True

        return False

    def circle_grid_coords(self, unit, radius):
        # Generates coordinates in grid that lie within a circle

        r = radius
        x_floor = int(unit.pos.x)
        y_floor = int(unit.pos.y)

        points = []
        for x in range(-r, r + 1):
            Y = int(math.sqrt(abs(r*r-x*x))) # bound for y given x
            for y in range(- Y, + Y + 1):
                points.append((x_floor + x, y_floor + y))
        return points

    def get_surrounding_points(self, unit, include_self=False):

        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma), (x, y - 2 * ma),
            (x + 2 * ma, y), (x - 2 * ma, y),
            (x + ma, y + ma), (x - ma, y - ma),
            (x + ma, y - ma), (x - ma, y + ma)
        ]

        if include_self:
            points.append((x, y))

        return points

    def check_bounds(self, x, y):
        return x >= 0 and y >=0 and x < self.map_x and y < self.map_y

    def get_surrounding_pathing(self, unit):

        points = self.get_surrounding_points(unit, include_self=False)
        vals = [self.pathing_grid[x, y] / 255 if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def get_surrounding_height(self, unit):

        points = self.get_surrounding_points(unit, include_self=True)
        vals = [self.terrain_height[x, y] / 255 if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def get_obs_agent(self, agent_id, global_obs=False):

        unit = self.get_unit_by_id(agent_id)

        move_feats = np.zeros(self.move_feats_len, dtype=np.float32) # exclude no-op & stop
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

                if dist < sight_range and e_unit.health > 0: # visible and alive

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

                    self.obs_set(category="distance_sc2", val=dist / sight_range, obs=enemy_feats[e_id], obs_type="enemy")
                    self.obs_set(category="x_sc2", val=(e_x - x) / sight_range, obs=enemy_feats[e_id], obs_type="enemy")
                    self.obs_set(category="y_sc2", val=(e_y - y) / sight_range, obs=enemy_feats[e_id], obs_type="enemy")

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="unit_global", val=self.get_unit_type_id_metamix(e_unit, False, self.min_unit_type) + 1, obs=enemy_feats[e_id], obs_type="enemy")

                    if self.obs_all_health:
                        self.obs_set(category="health", val=e_unit.health / e_unit.health_max, obs=enemy_feats[e_id],
                                     obs_type="enemy")

                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            self.obs_set(category="shield", val=e_unit.shield / max_shield, obs=enemy_feats[e_id],
                                         obs_type="enemy")

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        self.obs_set(category="unit_local", val=np.eye(self.unit_type_bits)[type_id], obs=enemy_feats[e_id], obs_type="enemy")

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
        
                if dist < sight_range and al_unit.health > 0: # visible and alive

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="id", val=al_id, obs=ally_feats[i],
                                     obs_type="ally")  # NEW: have to do this for grid-based input!

                    self.obs_set(category="visible", val=1, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="distance_sc2", val=dist / sight_range, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="x_sc2", val=(al_x - x) / sight_range, obs=ally_feats[i], obs_type="ally")
                    self.obs_set(category="y_sc2", val=(al_y - y) / sight_range, obs=ally_feats[i], obs_type="ally")

                    if self.obs_id_encoding == "metamix":
                        self.obs_set(category="unit_global", val=self.get_unit_type_id_metamix(al_unit, True, self.min_unit_type) + 1, obs=ally_feats[i], obs_type="ally")

                    if self.obs_all_health:
                        self.obs_set(category="health", val=al_unit.health / al_unit.health_max, obs=ally_feats[i],
                                     obs_type="ally")

                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            self.obs_set(category="shield", val=al_unit.shield / max_shield, obs=ally_feats[i],
                                         obs_type="ally")

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        self.obs_set(category="unit_local", val=np.eye(self.unit_type_bits)[type_id], obs=ally_feats[i], obs_type="ally")

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

    @staticmethod
    def _grid_rasterise(ally_feats, enemy_feats, obs_grid_shape, obs_get, obs_set, to_grid_coords, from_grid_coords, debug=False):
        # will store
        width, height = obs_grid_shape
        for al_id, al_feat in enumerate(ally_feats):
            if obs_get(category="visible", obs=al_feat, obs_type="ally") == 1.0:
                x_sc2 = obs_get(category="x_sc2", obs=al_feat, obs_type="ally")
                y_sc2 = obs_get(category="y_sc2", obs=al_feat, obs_type="ally")
                x, y = to_grid_coords(x_sc2, y_sc2)
                raster_x, raster_y = from_grid_coords(x, y)
                if not debug:
                    obs_set(category="distance_grid", val=(x**2 + y**2)**0.5, obs=al_feat, obs_type="ally")
                    obs_set(category="x_grid", val=x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_grid", val=y, obs=al_feat, obs_type="ally")
                    obs_set(category="distance_sc2_raster", val=(raster_x**2 + raster_y**2)**0.5, obs=al_feat, obs_type="ally")
                    obs_set(category="x_sc2_raster", val=raster_x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_sc2_raster", val=raster_y, obs=al_feat, obs_type="ally")
                else:
                    obs_set(category="distance_sc2", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=al_feat, obs_type="ally")
                    obs_set(category="x_sc2", val=raster_x, obs=al_feat, obs_type="ally")
                    obs_set(category="y_sc2", val=raster_y, obs=al_feat, obs_type="ally")

        for en_id, en_feat in enumerate(enemy_feats):
            if obs_get(category="visible", obs=en_feat, obs_type="enemy") == 1.0:
                x_sc2 = obs_get(category="x_sc2", obs=en_feat, obs_type="enemy")
                y_sc2 = obs_get(category="y_sc2", obs=en_feat, obs_type="enemy")
                x, y = to_grid_coords(x_sc2, y_sc2)
                raster_x, raster_y = from_grid_coords(x, y)
                if not debug:
                    obs_set(category="distance_grid", val=(x**2 + y**2)**0.5, obs=en_feat, obs_type="enemy")
                    obs_set(category="x_grid", val=x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_grid", val=y, obs=en_feat, obs_type="enemy")
                    obs_set(category="distance_sc2_raster", val=(raster_x**2 + raster_y**2)**0.5, obs=en_feat, obs_type="enemy")
                    obs_set(category="x_sc2_raster", val=raster_x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_sc2_raster", val=raster_y, obs=en_feat, obs_type="enemy")
                else:
                    obs_set(category="distance_sc2", val=(raster_x ** 2 + raster_y ** 2) ** 0.5, obs=en_feat, obs_type="enemy")
                    obs_set(category="x_sc2", val=raster_x, obs=en_feat, obs_type="enemy")
                    obs_set(category="y_sc2", val=raster_y, obs=en_feat, obs_type="enemy")

        return ally_feats, enemy_feats

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
    def _multiple_occupancy(ally_feats, enemy_feats, obs_grid_shape, obs_get, obs_set, to_grid_coords, from_grid_coords, debug):  # deal with multiple occupancy
        """
        Resolve multiple occupancy trouble from grid rep

        :param enemy_feats:
        :return:
        """
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

        #import numpy as np
        #dbg_grid = np.zeros((width, height))
        #for k, v in pos_hash.items():
        #    dbg_grid[int(k[0]), int(k[1])] = len(v)
        #print(dbg_grid)

        def _pref_sort(lst, obs_id_encoding = "metamix"):
            assert obs_id_encoding == "metamix", "not implemented!"
            # arrange elements such that the one least critical to be relocated is popped first
            allegiance_dict = {"ally":0, "enemy":1}
            type_dict = {1: 0, 2: 0, 3: 1} # prioritise long range units to be relocated (1: short range unit)
            try:
                tmp = sorted(lst, key=lambda row: type_dict[int(obs_get(category="unit_global", obs=row["feat"], obs_type="enemy"))], reverse=True)
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
                            x, y = int(coord[0]), int(coord[1])
                            dist = (x ** 2 + y ** 2) ** 0.5
                            x_raster, y_raster = from_grid_coords(coord[0],
                                                                  coord[1])
                            dist_raster = (x_raster ** 2 + y_raster ** 2) ** 0.5
                            feats = ally_feats if unit["utype"] == "ally" else enemy_feats
                            obs_set(category="distance_sc2_raster", val=dist_raster, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="x_sc2_raster", val=x_raster, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="y_sc2_raster", val=y_raster, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="distance_grid", val=dist, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="x_grid", val=x, obs=feats[unit["id"]], obs_type=unit["utype"])
                            obs_set(category="y_grid", val=y, obs=feats[unit["id"]], obs_type=unit["utype"])
                        break

        #import numpy as np
        #dbg_grid = np.zeros((width, height))
        #for k, v in pos_hash.items():
        #    dbg_grid[int(k[0]), int(k[1])] = len(v)
        #print(dbg_grid)

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

        ally_feats_batch = ally_feats.contiguous().view(-1, n_agents-1, ally_feats.shape[-1]//(n_agents-1))
        enemy_feats_batch = enemy_feats.contiguous().view(-1, n_enemies, enemy_feats.shape[-1]//n_enemies)

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
        obs_dict["move"] = obs[...,ctr:ctr + move_feats_len]
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
                grid_flat = (unit_global-1)*obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                unit_global = obs_get(category="unit_global", obs=feat, obs_type="enemy")
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                grid_flat = (unit_global-1) * obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
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
                grid_flat = obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
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
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible*unit_id)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                unit_id = obs_get(category="id", obs=feat, obs_type="enemy") + 1
                grid_flat = obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible*unit_id)
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
                grid_flat_x = 0*obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                grid_flat_y = 1*obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat_x.long(), visible*x_sc2)
                channels_flat_grid.scatter_(-1, grid_flat_y.long(), visible*y_sc2)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                x_sc2 = obs_get(category="x_sc2", obs=feat, obs_type="enemy")
                y_sc2 = obs_get(category="y_sc2", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                grid_flat_x = 2*obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                grid_flat_y = 3*obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat_x.long(), visible*x_sc2)
                channels_flat_grid.scatter_(-1, grid_flat_y.long(), visible*y_sc2)

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
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible*health)

            for _id in range(obs_dict["enemies"].shape[-2]):
                feat = obs_dict["enemies"][..., _id, :]
                x_grid = obs_get(category="x_grid", obs=feat, obs_type="enemy")
                y_grid = obs_get(category="y_grid", obs=feat, obs_type="enemy")
                visible = obs_get(category="visible", obs=feat, obs_type="enemy")
                health = obs_get(category="health", obs=feat, obs_type="ally")
                grid_flat = obs_grid_shape[0]*obs_grid_shape[1] + y_grid * obs_grid_shape[0] + x_grid
                channels_flat_grid.scatter_(-1, grid_flat.long(), visible*health)
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
        cache["grid_coords_flat"] = th.cat([grid_coords_flat_ally, grid_coords_flat_enemy], dim=-2).view(*grid_coords_flat_ally.shape[:-2],
                                                                                                         -1)
        cache["x_grid"] = th.cat([cache["x_grid_ally"], cache["x_grid_enemy"]], dim=-2)
        cache["y_grid"] = th.cat([cache["y_grid_ally"], cache["y_grid_enemy"]], dim=-2)
        cache["y_grid"] = th.cat([cache["y_grid_ally"], cache["y_grid_enemy"]], dim=-2)
        cache["visible"] = th.cat([cache["visible_ally"], cache["visible_enemy"]], dim=-2).view(*cache["visible_ally"].shape[:-2],
                                                                                                         -1)
        cache["ally_enemy_mask"] = cache["x_grid"].clone().zero_()
        cache["ally_enemy_mask"][..., obs_dict["allies"].shape[-2]:, :] = 1 # enemies are 1 in this mask, allies zero
        cache["ally_enemy_mask"] = cache["ally_enemy_mask"].view(*cache["ally_enemy_mask"].shape[:-2],
                                                                 -1)

        # calculate and assign empty memory for all channels
        dim_channels_dict = {"unit_type": {"uncompressed":10, "compressed":10},
                             "ally_or_enemy": {"uncompressed":2, "compressed":1},
                             "id": {"uncompressed":2, "compressed":1},
                             "precise_pos": {"uncompressed":4, "compressed":2},
                             "health": {"uncompressed":2, "compressed":1},
                             "last_action": {"uncompressed":2, "compressed":2},
                             "attackable": {"uncompressed":1, "compressed":1}}

        n_channels = sum([dim_channels_dict[label["type"]][label.get("mode", "uncompressed")] for label in label_lst])

        channels_slice_dict = {}
        ctr = 0
        for label in label_lst:
            channels_slice_dict[label["type"]] = slice(ctr, ctr + dim_channels_dict[label["type"]][label.get("mode", "uncompressed")])
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
                        cache["unit_global_allies"] = obs_get(category="unit_global", obs=obs_dict["allies"], obs_type="ally").long()
                    unit_global_allies = cache["unit_global_allies"]

                    if not ("unit_global_enemies" in cache):
                        cache["unit_global_enemies"] = obs_get(category="unit_global", obs=obs_dict["enemies"], obs_type="enemy").long()
                    unit_global_enemies = cache["unit_global_enemies"]

                    cache["unit_global"] = th.cat([unit_global_allies, unit_global_enemies], dim=-2)
                    cache["unit_global"] = cache["unit_global"].view(*cache["unit_global"].shape[:-2],
                                                                     -1)

                unit_global = cache["unit_global"]
                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]

                # TODO: use diverse label types for units!
                gcf = (label_slice.start + unit_global)*obs_grid_shape[0]*obs_grid_shape[1] + grid_coords_flat
                channels_flat_grid.scatter_(-1, gcf, visible)

            elif label_type == "ally_or_enemy":
                visible = cache["visible"]
                grid_coords_flat = cache["grid_coords_flat"]
                ally_enemy_mask = cache["ally_enemy_mask"]

                if label_mode == "compressed":
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible*(ally_enemy_mask*2-1))
                elif label_mode == "uncompressed":
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[1]
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
                    channels_flat_grid.scatter_(-1, gcf, visible*((1-ally_enemy_mask)*2-1)*unit_id)

                elif label_mode == "uncompressed":
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible*unit_id)

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
                    gcf = grid_coords_flat +  label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * x_sc2)
                    gcf = grid_coords_flat +  1 * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * y_sc2)

                elif label_mode == "uncompressed":
                    ally_enemy_mask = cache["ally_enemy_mask"]
                    gcf = grid_coords_flat +  (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * x_sc2)
                    gcf = grid_coords_flat +  2 * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * y_sc2)

            elif label_type == "last_action":

                if not ("last_action" in cache):
                    cache["last_action_no_attack"] = obs_get(category="last_action", obs=obs_dict["allies"], obs_type="ally")

                    cache["last_action_no_attack"] = cache["last_action_no_attack"].view(*grid_coords_flat_ally.shape[:-2],
                                                             -1)

                if not ("last_attacked_grid" in cache):
                    cache["last_attacked_grid"] = obs_get(category="health", obs=obs_dict["allies"], obs_type="ally")
                    cache["last_attacked_grid"] = cache["last_attacked_grid"].view(*grid_coords_flat_ally.shape[:-2],
                                                             -1)

                grid_coords_flat = cache["grid_coords_flat"]
                visible = cache["visible"]
                last_action_no_attack = cache["last_action_no_attack"]
                last_attacked_grid = cache["last_attacked_grid"]

                if label_mode in ["compressed","uncompressed"]:
                    gcf = grid_coords_flat + label_slice.start * obs_grid_shape[0] * obs_grid_shape[1]
                    channels_flat_grid.scatter_(-1, gcf, visible * last_action_no_attack)
                    channels[..., label_slice.start+1 , :, :] = last_attacked_grid

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
                    gcf = grid_coords_flat + (label_slice.start + ally_enemy_mask) * obs_grid_shape[0] * obs_grid_shape[1]
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

            label_lst = [{"type":"ally_or_enemy", "mode":"compressed"},
                         {"type":"id", "mode":"compressed"},
                         {"type":"health", "mode":"compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__3m_noid":

            label_lst = [{"type":"ally_or_enemy", "mode":"compressed"},
                         {"type":"last_action", "mode":"compressed"},
                         {"type":"multiple_occupancy", "mode":"compressed"},
                         {"type":"health", "mode":"compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__3m_precise":
            label_lst = [{"type":"ally_or_enemy", "mode":"compressed"},
                         {"type":"id", "mode":"compressed"},
                         {"type":"health", "mode":"compressed"},
                         {"type":"precise_pos", "mode":"compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            return channels

        elif scenario == "metamix__2s3z":
            label_lst = [{"type":"unit_type", "mode":"compressed"},
                         {"type":"ally_or_enemy", "mode":"compressed"},
                         {"type":"id", "mode":"compressed"},
                         {"type":"health", "mode":"compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            channels_out = th.cat([channels[..., 1:3, :, :],
                                   channels[..., 10:, :, :]], dim=-3) # remove superfluous type channels
            return channels_out

        elif scenario == "metamix__2s3z_noid":
            label_lst = [{"type":"unit_type", "mode":"compressed"},
                         {"type":"ally_or_enemy", "mode":"compressed"},
                         # {"type":"id", "mode":"compressed"},
                         {"type":"last_action", "mode":"compressed"},
                         #{"type":"multiple_occupancy", "mode":"compressed"},
                         {"type":"health", "mode":"compressed"}]
            channels = create_channels(label_lst, obs, get_size_only=get_size_only)
            channels_out = th.cat([channels[..., 1:3, :, :],
                                   channels[..., 10:, :, :]], dim=-3) # remove superfluous type channels
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

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):

        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_id)

                ally_state[al_id, 0] = al_unit.health / al_unit.health_max # health
                if self.map_type == 'MMM' and al_unit.unit_type == self.medivac_id:
                    ally_state[al_id, 1] = al_unit.energy / max_cd # energy
                else:
                    ally_state[al_id, 1] = al_unit.weapon_cooldown / max_cd # cooldown
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x # relative X
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = al_unit.shield / max_shield # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = e_unit.health / e_unit.health_max # health
                enemy_state[e_id, 1] = (x - center_x) / self.max_distance_x # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.max_distance_y # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = e_unit.shield / max_shield # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        state = state.astype(dtype=np.float32)

        if self.debug_inputs:
            print("------------ STATE ---------------")
            print("Ally state\n", ally_state)
            print("Enemy state\n", enemy_state)
            print("Last action\n", self.last_action)
            print("----------------------------------")

        return state

    @staticmethod
    def get_unit_type_id_metamix(unit, ally, min_unit_type):

        if ally: # we use new SC2 unit types
            type_id = unit.unit_type - min_unit_type

        else: # 'We use default SC2 unit types'

            # if self.map_type == 'sz':
            #     # id(Stalker) = 74, id(Zealot) = 73
            #     type_id = unit.unit_type - 73
            # if self.map_type == 'ze_ba':
            #     if unit.unit_type == 9:
            #         type_id = 0
            #     else:
            #         type_id = 1
            # elif self.map_type == 'MMM':
            #     if unit.unit_type == 51:
            #         type_id = 0
            #     elif unit.unit_type == 48:
            #         type_id = 1
            #     else:
            #         type_id = 2
            # if self.map_type == 'marines':
            #     type_id = 0

            # unit ids coincide with self-made units
            unit_id_dict = {48: 0, # marines
                            74: 1, # stalkers
                            73: 2,  # zealots
                            }
            # TODO: extend for marines, stalkers, zealots, marauder, medivac, spine crawler, baneling, zergling, hydralisk, colossus
            type_id = unit_id_dict[unit.unit_type]

        return type_id

    def get_unit_type_id(self, unit, ally):

        if ally: # we use new SC2 unit types
            type_id = unit.unit_type - self.min_unit_type

        else: # 'We use default SC2 unit types'

            if self.map_type == 'sz':
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            if self.map_type == 'ze_ba':
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == 'MMM':
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2

        return type_id

    def get_state_size(self):

        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions

        return size

    def get_avail_agent_actions(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot do no-op as alife
            avail_actions = [0] * self.n_available_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, 0):
                avail_actions[2] = 1
            if self.can_move(unit, 1):
                avail_actions[3] = 1
            if self.can_move(unit, 2):
                avail_actions[4] = 1
            if self.can_move(unit, 3):
                avail_actions[5] = 1

            # can attack only those who are alive
            # and in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()
            if self.map_type == 'MMM' and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves and other flying units
                target_items = [(t_id, t_unit) for (t_id, t_unit) in self.agents.items() if t_unit.unit_type != self.medivac_id]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_unrestricted_actions(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot do no-op as alife
            avail_actions = [1] * self.n_actions
            avail_actions[0] = 0
        else:
            avail_actions = [0] * self.n_actions
            avail_actions[0] = 1
        return avail_actions

    def get_avail_actions(self):

        avail_actions = []
        for agent_id in range(self.n_agents):
            if self.restrict_actions:
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            else:
                avail_agent = self.get_unrestricted_actions(agent_id)
                avail_actions.append(avail_agent)

        return avail_actions

    def get_obs_size(self):

        # if self.grid_obs:
        #     if not self.grid_obs_compressed:
        #         return self.n_obs_channels * self.grid_width * self.grid_height + self.own_feats_size + self.move_feats_size
        #     else:
        #         return  self.ally_feats_size + self.enemy_feats_size + self.move_feats_size + self.own_feats_size
        # else:
        if not self.obs_decode_on_the_fly:
            assert False, "not implemented"

        return self.obs_size

    def close(self):
        self._sc2_proc.close()

    def render(self):
        pass

    def kill_all_units(self):

        units_alive = [unit.tag for unit in self.agents.values() if unit.health > 0] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        debug_command = [d_pb.DebugCommand(kill_unit = d_pb.DebugKillUnit(tag = units_alive))]
        self.controller.debug(debug_command)

    def init_units(self):

        # In case controller step fails
        while True:

            self.agents = {}
            self.enemies = {}

            ally_units = [unit for unit in self._obs.observation.raw_data.units if unit.owner == 1]
            ally_units_sorted = sorted(ally_units, key=attrgetter('unit_type', 'pos.x', 'pos.y'), reverse=False)

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug_inputs:
                    print("Unit %d is %d, x = %.1f, y = %1.f"  % (len(self.agents), self.agents[i].unit_type, self.agents[i].pos.x, self.agents[i].pos.y))

            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 1:
                        self.max_reward += unit.health_max + unit.shield_max

            if self._episode_count == 1:
                min_unit_type = min(unit.unit_type for unit in self.agents.values())
                self.init_ally_unit_types(min_unit_type)

            if len(self.agents) == self.n_agents and len(self.enemies) == self.n_enemies:
                # All good
                return

            try:
                self.controller.step(1)
                self._obs = self.controller.observe()
            except protocol.ProtocolError:
                self.full_restart()
                self.reset()
            except protocol.ConnectionError:
                self.full_restart()
                self.reset()

    def update_units(self):

        # This function assumes that self._obs is up-to-date
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_agent_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated: # means dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated: # means dead
                e_unit.health = 0

        if self.heuristic:
            for al_id, al_unit in self.agents.items():
                current_target = self.heuristic_targets[al_id]
                if current_target == 0 or self.enemies[current_target].health == 0:
                    x = al_unit.pos.x
                    y = al_unit.pos.y
                    min_dist = 32
                    min_id = -1
                    for e_id, e_unit in self.enemies.items():
                        if e_unit.health > 0:
                            dist = self.distance(x, y, e_unit.pos.x, e_unit.pos.y)
                            if dist < min_dist:
                                min_dist = dist
                                min_id = e_id
                    self.heuristic_targets[al_id] = min_id

        if (n_ally_alive == 0 and n_enemy_alive > 0) or self.only_medivac_left(ally=True):
            return -1 # loss
        if (n_ally_alive > 0 and n_enemy_alive == 0) or self.only_medivac_left(ally=False):
            return 1 # win
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0 # tie, not sure if this is possible

        return None

    def only_medivac_left(self, ally):
        if self.map_type != 'MMM':
            return False

        if ally:
            units_alive = [a for a in self.agents.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [a for a in self.enemies.values() if (a.health > 0 and a.unit_type != self.medivac_id)]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        return self.agents[a_id]

    def get_stats(self):
        stats = {}
        stats["battles_won"] = self.battles_won
        stats["battles_game"] = self.battles_game
        stats["battles_draw"] = self.timeouts
        stats["win_rate"] = self.battles_won / self.battles_game
        stats["timeouts"] = self.timeouts
        stats["restarts"] = self.force_restarts
        return stats

    def get_agg_stats(self, stats):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean([ (_a - _b)/(_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                                              [0]*len(current_stats["battles_won"]) if self.last_stats is None else self.last_stats["battles_won"],
                                                                                              current_stats["battles_game"],
                                                                                              [0]*len(current_stats["battles_game"]) if self.last_stats is None else
                                                                                              self.last_stats["battles_game"])
                                                if (_c - _d) != 0.0])
            else:
                aggregate_stats[_k] = np.mean([_a-_b for _a, _b in zip(_v, [0]*len(_v) if self.last_stats is None else self.last_stats[_k])])

        self.last_stats = current_stats
        return aggregate_stats

    @staticmethod
    def state_decoder(state): # decode obs tensor into real shape
        return state

    @staticmethod
    def grid_obs_decoder(obs,
                         n_channels,
                         width,
                         height,
                         ally_feat_size,
                         enemy_feat_size,
                         move_feat_size,
                         own_feat_size,
                         compressed,
                         obs_explode,
                         obs_get,
                         decompressor=None,
                         device=None):  # decode obs tensor into real shape

        if obs is None:
            return {"2d": [(n_channels,
                            width,
                            height), (1, width, height)],
                    "1d": [move_feat_size + own_feat_size, 1, 1]}
        else:
            if not compressed:
                lst_2d = [slice(None)] * (obs.dim() - 1) + [slice(0, obs.shape[-1] - move_feat_size - own_feat_size)]
                lst_1d = [slice(None)] * (obs.dim() - 1) + [slice(obs.shape[-1] - move_feat_size - own_feat_size, None)]
                return {"2d": [obs[lst_2d].view(*obs.shape[:-1],
                                                n_channels,
                                                width,
                                                height), ],
                        "1d": [obs[lst_1d]]}
            else:  # uncompress!

                import torch as th
                grid = obs.new(*obs.shape[:-1],
                               n_channels,
                               width,
                               height)
                if device is not None:
                    grid.to(device)


                obs_dict = obs_explode(obs)
                enemy_grid_x = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
                enemy_grid_y = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
                visible = obs_get(category="visible", obs=obs_dict["enemies"], obs_type="enemy")

                grid = decompressor(obs)

                lst_1d = [slice(None)] * (obs.dim() - 1) + [slice(obs.shape[-1] - move_feat_size - own_feat_size, None)]
                return {"2d": [grid.to(obs.device)],
                        "1d": [obs[lst_1d].to(obs.device), enemy_grid_x, enemy_grid_y, visible]}

    @staticmethod
    def transformer_obs_decoder(obs,
                    move_feat_size,
                    own_feat_size,
                    n_agents,
                    nf_al,
                    n_enemies,
                    nf_en):  # decode obs tensor into real shape

        if obs is None:
            return {"2d": [(n_agents-1, nf_al), (n_enemies, nf_en)],
                    "1d": [move_feat_size + own_feat_size]}
        else:
            al_2d = [slice(None)] * (obs.dim() - 1) + [slice(0, (n_agents-1)*nf_al)]
            en_2d = [slice(None)] * (obs.dim() - 1) + [slice((n_agents-1)*nf_al, (n_agents-1)*nf_al + n_enemies*nf_en)]
            move_1d = [slice(None)] * (obs.dim() - 1) + [slice((n_agents-1)*nf_al + n_enemies*nf_en,
                                                               (n_agents-1)*nf_al + n_enemies*nf_en + move_feat_size)]
            own_feat_1d = [slice(None)] * (obs.dim() - 1) + [slice((n_agents-1) * nf_al + n_enemies*nf_en + move_feat_size,
                                                                   (n_agents-1) * nf_al + n_enemies * nf_en + move_feat_size + own_feat_size)]
            features = {"2d": [obs[al_2d].view(*obs.shape[:-1], (n_agents-1), nf_al).to(obs.device),
                               obs[en_2d].view(*obs.shape[:-1], n_enemies, nf_en).to(obs.device)],
                        "1d": [th.cat([obs[move_1d].to(obs.device),
                                       obs[own_feat_1d].to(obs.device)], dim=-1)]
                        }
            return features

    @staticmethod
    def actions_decoder_grid(actions_flat,
                             avail_actions,
                             obs,
                             obs_grid_shape,
                             n_actions_no_attack,
                             obs_get,
                             obs_explode,
                             mask_val=float("-9999999")):  # decode obs tensor into real shape
        """
        decodes elements of the encoding space into valid elements of the env action space
        for this purpose, need to pass both obs and avail_actions

        this is being used to e.g. convert grid q-values from grid output to valid actions,
        in particular attack actions
        """

        actions = avail_actions.clone().float().zero_()

        # no attack actions require no special decoding
        actions[..., :n_actions_no_attack] = grid_actions[..., :n_actions_no_attack]

        # handle attack actions
        obs_dict = obs_explode(obs)
        x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        id_enemy = obs_get(category="id", obs=obs_dict["enemies"], obs_type="enemy").long()
        id_enemy = id_enemy.view(*id_enemy.shape[:-2], -1)
        visible = obs_get(category="visible", obs=obs_dict["enemies"], obs_type="enemy")
        visible = visible.view(*visible.shape[:-2], -1)

        grid_flat_enemy = y_grid_enemy * obs_grid_shape[0] + x_grid_enemy
        grid_flat_enemy = grid_flat_enemy.view(*grid_flat_enemy.shape[:-2], -1)
        grid_actions_attack = grid_actions[..., n_actions_no_attack:] # .contiguous().view(*grid_actions.shape[:-2], -1)

        mask = visible
        vals = grid_actions_attack.gather(-1, grid_flat_enemy) * mask + mask_val * (1.0 - mask)

        # if id_enemy is zero, no damage is created by overwriting first enemy attack position!
        actions.scatter_(-1, id_enemy + n_actions_no_attack, vals)

        # very important: apply avail actions at the end!
        out_actions = actions * avail_actions.float() + (1-avail_actions.float()) * mask_val
        return out_actions

    @staticmethod
    def actions_encoder_grid(actions_sc2,
                             obs,
                             n_actions_no_attack,
                             obs_grid_shape,
                             obs_get,
                             obs_explode):  # decode obs tensor into real shape
        """
        turns SC2 env actions into grid actions
        """
        actions = th.zeros([*actions_sc2.shape[:-1], 2, obs_grid_shape[0], obs_grid_shape[1]],
                           dtype=th.float32,
                           device=actions_sc2.device)
        obs_dict = obs_explode(obs)
        x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        grid_flat_enemy = (y_grid_enemy * obs_grid_shape[0] + x_grid_enemy.long())

        # mask all actions that actually correspond to attack actions
        action_is_attack = (actions_sc2 >= n_actions_no_attack)

        # select the flat grid indices that correspond to the agents that are being attacked
        action_enemy_idx = grid_flat_enemy.squeeze(-1).gather(-1, ((actions_sc2 - n_actions_no_attack) * action_is_attack.long()))

        # fill in the attack actions
        actions[..., 1, :, :].view(*actions.shape[:-3], -1).scatter_(-1, action_enemy_idx, action_is_attack.float())

        # now fill in the other attack actions with fixed action grids

        # action 0: leave all-zeros (no action)

        # action 1: fill in middle tile (stop action)
        actions[..., 0, obs_grid_shape[0]//2, obs_grid_shape[1]//2][(actions_sc2==1).squeeze()] = 1.0

        # action 2: fill in the north tile (north)
        actions[..., 0, obs_grid_shape[0] // 2, obs_grid_shape[1] // 2 - 1][(actions_sc2 == 2).squeeze()] = 1.0

        # action 3: fill in the south tile (south)
        actions[..., 0, obs_grid_shape[0] // 2, obs_grid_shape[1] // 2 + 1][(actions_sc2 == 3).squeeze()] = 1.0

        # action 4: fill in the east tile (east)
        actions[..., 0, obs_grid_shape[0] // 2 + 1, obs_grid_shape[1] // 2][(actions_sc2 == 4).squeeze()] = 1.0

        # action 5: fill in the west tile (west)
        actions[..., 0, obs_grid_shape[0] // 2 - 1, obs_grid_shape[1] // 2][(actions_sc2 == 5).squeeze()] = 1.0

        return actions

        #actions[..., 1, :, :].view(*actions.shape[:-3], -1).scatter_(-1, grid_flat_enemy, action_enemy_idx.float())
        #actions[..., 0, :, :].view(*actions.shape[:-3], -1).scatter_(-1, grid_flat_enemy, 1.0 - action_is_attack)

        # actions = avail_actions.clone().float().zero_()
        #
        # # no attack actions require no special decoding
        # actions[..., :n_actions_no_attack] = grid_actions[..., :n_actions_no_attack]
        #
        # # handle attack actions
        # obs_dict = obs_explode(obs)
        # x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        # y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        # id_enemy = obs_get(category="id", obs=obs_dict["enemies"], obs_type="enemy").long()
        # id_enemy = id_enemy.view(*id_enemy.shape[:-2], -1)
        # visible = obs_get(category="visible", obs=obs_dict["enemies"], obs_type="enemy")
        # visible = visible.view(*visible.shape[:-2], -1)
        #
        # grid_flat_enemy = y_grid_enemy * obs_grid_shape[0] + x_grid_enemy
        # grid_flat_enemy = grid_flat_enemy.view(*grid_flat_enemy.shape[:-2], -1)
        # grid_actions_attack = grid_actions[..., n_actions_no_attack:] # .contiguous().view(*grid_actions.shape[:-2], -1)
        #
        # mask = visible
        # vals = grid_actions_attack.gather(-1, grid_flat_enemy) * mask + mask_val * (1.0 - mask)
        #
        # # if id_enemy is zero, no damage is created by overwriting first enemy attack position!
        # actions.scatter_(-1, id_enemy + n_actions_no_attack, vals)
        #
        # # very important: apply avail actions at the end!
        # out_actions = actions * avail_actions.float() + (1-avail_actions.float()) * mask_val
        # return out_actions


    @staticmethod
    def avail_actions_encoder_grid(avail_actions,
                                   obs,
                                   obs_grid_shape,
                                   n_actions_no_attack,
                                   obs_get,
                                   obs_explode,
                                   mask_val=float("-9999999")):  # decode obs tensor into real shape
        """
        turn SC2 env avail actions into grid representation
        """
        # Step 1: Turn each action available into an actual SC2 action
        #actions = avail_actions.clone()
        mask = th.arange(0,
                         avail_actions.shape[-1],
                         dtype=th.int32,
                         device=avail_actions.device).expand_as(avail_actions)
        actions_sc2 = (avail_actions * mask).unsqueeze(-1)

        # Step 2: proceed as with action encoding
        actions = th.zeros([*actions_sc2.shape[:-1], 2, obs_grid_shape[0], obs_grid_shape[1]],
                           dtype=th.float32,
                           device=actions_sc2.device)
        obs_dict = obs_explode(obs)
        x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemies"], obs_type="enemy").long()
        grid_flat_enemy = (y_grid_enemy * obs_grid_shape[0] + x_grid_enemy.long())

        # mask all actions that actually correspond to attack actions
        action_is_attack = (actions_sc2 >= n_actions_no_attack)

        # select the flat grid indices that correspond to the agents that are being attacked
        grid_flat = grid_flat_enemy.squeeze(-1).unsqueeze(-2)
        action_enemy_idx = grid_flat.expand(*grid_flat.shape[:-2],
                                            actions_sc2.shape[-2],
                                            grid_flat.shape[-1]
                                            ).gather(-1, ((actions_sc2 - n_actions_no_attack).long() * action_is_attack.long()))

        # fill in the attack actions
        actions[..., 1, :, :].view(*actions.shape[:-3], -1).scatter_(-1, action_enemy_idx, action_is_attack.float())

        # now fill in the other attack actions with fixed action grids

        # action 0: leave all-zeros (no action)

        # action 1: fill in middle tile (stop action)
        actions[..., 0, obs_grid_shape[0]//2, obs_grid_shape[1]//2][(actions_sc2==1).squeeze()] = 1.0

        # action 2: fill in the north tile (north)
        actions[..., 0, obs_grid_shape[0] // 2, obs_grid_shape[1] // 2 - 1][(actions_sc2 == 2).squeeze()] = 1.0

        # action 3: fill in the south tile (south)
        actions[..., 0, obs_grid_shape[0] // 2, obs_grid_shape[1] // 2 + 1][(actions_sc2 == 3).squeeze()] = 1.0

        # action 4: fill in the east tile (east)
        actions[..., 0, obs_grid_shape[0] // 2 + 1, obs_grid_shape[1] // 2][(actions_sc2 == 4).squeeze()] = 1.0

        # action 5: fill in the west tile (west)
        actions[..., 0, obs_grid_shape[0] // 2 - 1, obs_grid_shape[1] // 2][(actions_sc2 == 5).squeeze()] = 1.0

        return actions


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
    def actions_encoder(actions,
                        avail_actions,
                        obs,
                        obs_grid_shape,
                        obs_get,
                        width,
                        height,
                        ally_feat_size,
                        enemy_feat_size,
                        move_feat_size,
                        own_feat_size,
                        n_actions_no_attack):  # decode obs tensor into real shape
        """
        this function splits a given SC2 env action into two parts comprising an overall fixed size:
            - a fixed-size 1d part (non-attack actions)
            - a grid representation where 1 indicates position of enemy that can be attacked (attack actions)

        input additionally requires SC2 avail actions and obs
        """

        assert False, "Not implemented!"

        # grid = avail_actions.new(*avail_actions.shape[:-1], *obs_grid_shape).zero_()

        # grid_flat = grid.view(-1, width, height)

        # avail_actions_non_attack_ids = [slice(None)] * (len(avail_actions.shape) - 1) + [slice(0,
        #                                                                                 n_actions_no_attack,
        #                                                                                 None)]

        # avail_actions_attack_ids = [slice(None)] * (len(avail_actions.shape) - 1) + [slice(n_actions_no_attack,
        #                                                                                   None,
        #                                                                                    None)]
        # non_attack_avail_actions = avail_actions[..., :n_actions_no_attack]
        # attack_avail_actions = avail_actions[..., n_actions_no_attack:]
        #
        #
        # grid_flat[:, grid_x.view(-1), grid_y.view(-1)] = attack_avail_actions.view(-1, attack_avail_actions.shape[-1]) # todo: check this seriously
        # obs_dict = obs_explode(obs)
        # x_grid_enemy = obs_get(category="x_grid", obs=obs_dict["enemy"], obs_type="enemy").long()
        # y_grid_enemy = obs_get(category="y_grid", obs=obs_dict["enemy"], obs_type="enemy").long()
        # id_enemy = obs_get(category="id", obs=obs_dict["enemy"], obs_type="enemy").long()
        # visible = obs_get(category="visible", obs=obs_dict["enemy"], obs_type="enemy")
        #
        # return {"2d": [grid, ],
        #         "1d": [non_attack_avail_actions, ]}

        return

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
            n_channels, grid_width, grid_height = decompressor(th.zeros((1,1, self.get_obs_size())), get_size_only=True).shape[-3:]
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
                    #"actions_encoder": actions_encoder,
                    "actions_decoder_grid": actions_decoder_grid,
                    "actions_encoder_grid": actions_encoder_grid,
                    "avail_actions_encoder_grid": avail_actions_encoder_grid,
                    # "avail_actions_decoder_flat": avail_actions_decoder_flat,
                    "n_actions": self.n_actions_encoding, #self.get_total_actions(),
                    "n_actions_no_attack": self.n_actions_no_attack,
                    "n_available_actions": self.n_available_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

# from components.transforms_old import _seq_mean

class StatsAggregator():

    def __init__(self):
        self.last_stats = None
        self.stats = []
        pass

    def aggregate(self, stats, add_stat_fn):

        current_stats = {}
        for stat in stats:
            for _k, _v in stat.items():
                if not (_k in current_stats):
                    current_stats[_k] = []
                if _k in ["win_rate"]:
                    continue
                current_stats[_k].append(_v)

        # average over stats
        aggregate_stats = {}
        for _k, _v in current_stats.items():
            if _k in ["win_rate"]:
                aggregate_stats[_k] = np.mean([ (_a - _b)/(_c - _d) for _a, _b, _c, _d in zip(current_stats["battles_won"],
                                                                                              [0]*len(current_stats["battles_won"]) if self.last_stats is None else self.last_stats["battles_won"],
                                                                                              current_stats["battles_game"],
                                                                                              [0]*len(current_stats["battles_game"]) if self.last_stats is None else
                                                                                              self.last_stats["battles_game"])
                                                if (_c - _d) != 0.0])
            else:
                aggregate_stats[_k] = np.mean([_a-_b for _a, _b in zip(_v, [0]*len(_v) if self.last_stats is None else self.last_stats[_k])])

        # add stats that have just been produced to tensorboard / sacred
        for _k, _v in aggregate_stats.items():
            add_stat_fn(_k, _v)

        # collect stats for logging horizon
        self.stats.append(aggregate_stats)
        # update last stats
        self.last_stats = current_stats
        pass

    def log(self, log_directly=False):
        assert not log_directly, "log_directly not supported."
        logging_str = " Win rate: {}".format(_seq_mean([ stat["win_rate"] for stat in self.stats ]))\
                    + " Timeouts: {}".format(_seq_mean([ stat["timeouts"] for stat in self.stats ]))\
                    + " Restarts: {}".format(_seq_mean([ stat["restarts"] for stat in self.stats ]))

        # flush stats
        self.stats = []
        return logging_str

