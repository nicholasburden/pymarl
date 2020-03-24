import datetime
import os
import pprint
import time
import threading
import torch as th
import dill
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import collections
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

import yaml
def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    if args.meta == "reptile":
        run_reptile(args=args, logger=logger, _log=_log, _run=_run)

    else:
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_decoder = dill.loads(env_info["obs_decoder"]) if env_info["obs_decoder"] is not None else None
    args.avail_actions_encoder = dill.loads(env_info["avail_actions_encoder_grid"]) if env_info["avail_actions_encoder_grid"] is not None else None

    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents", "vshape_decoded": env_info.get("obs_shape_decoded", env_info["obs_shape"])},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))




    while runner.t_env <= args.t_max:
        th.cuda.empty_cache()
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        del episode_batch

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)



            learner.train(episode_sample, runner.t_env, episode)
            th.cuda.empty_cache()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            th.cuda.empty_cache()

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run
        th.cuda.empty_cache()

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def update_env(args, logger):
    temp = args.env_args["map_name"]
    args.env_args["map_name"] = args.env_args["map_name2"]
    args.env_args["map_name2"] = temp
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_decoder = dill.loads(env_info["obs_decoder"]) if env_info["obs_decoder"] is not None else None
    args.avail_actions_encoder = dill.loads(env_info["avail_actions_encoder_grid"]) if env_info[
                                                                                           "avail_actions_encoder_grid"] is not None else None

    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents",
                "vshape_decoded": env_info.get("obs_shape_decoded", env_info["obs_shape"])},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
    return runner, buffer, learner

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def run_reptile(args, logger, _log, _run):

    loggers = {}
    runners = {}
    macs = {}
    learners = {}
    buffers = {}

    agent_state_dict = None

    import yaml
    #from .main import _get_config
    # compile all the relevant task configs
    task_configs = {}

    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    r = np.random.RandomState(args.seed)
    for k, v in sorted(args.tasks.items()): # important for reproducibility of seeds!

        # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)

        # Load algorithm and env base configs
        params = ["", "--config={}".format(v.pop("config")), "--env-config={}".format(v.pop("env-config"))]
        alg_config = _get_config(params, "--config", "algs")
        env_config = _get_config(params, "--env-config", "envs")

        # config_dict = {**config_dict, **env_config, **alg_config}
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)
        config_dict = recursive_dict_update(config_dict, v)

        # from src.utils.dict2namedtuple import convert
        config_dict.pop("no-mongo")
        config_dict["seed"] = r.randint(0, 2**31-1) # have to set manually
        config_dict["env_args"]["seed"] = r.randint(0, 2**31-1)
        config_dict["device"] = args.device
        config_dict["unique_token"] = "{}__{}".format(args.unique_token,
                                                     k)
        task_configs[k] = Bunch(config_dict)

    def setup_components(logger,
                         agent_state_dict):
        task_names = []
        for task_name, _ in task_configs.items():
            task_names.append(task_name)

        # set up tasks based on the configs
        for task_name, task_config in task_configs.items():

            task_args = task_config

            from copy import deepcopy
            logger = Logger(_log)
            # sacred is on by default
            logger.setup_sacred(_run)
            # logger = deepcopy(meta_logger)
            logger.prefix = task_name
            loggers[task_name] = logger

            # Init runner so we can get env info
            runner = r_REGISTRY[task_args.runner](args=task_args,
                                                  logger=logger)
            runners[task_name] = runner

            # Set up schemes and groups here
            env_info = runner.get_env_info()
            task_args.n_agents = env_info["n_agents"]
            task_args.n_actions = env_info["n_actions"]
            task_args.obs_decoder = dill.loads(env_info["obs_decoder"]) if env_info["obs_decoder"] is not None else None
            task_args.avail_actions_encoder = dill.loads(env_info["avail_actions_encoder_grid"]) if env_info[
                                                                                                   "avail_actions_encoder_grid"] is not None else None
            task_args.db_url = args.db_url
            task_args.db_name = args.db_name
            task_args.state_shape = env_info["state_shape"]
            task_args.state_decoder = dill.loads(env_info["state_decoder"]) if env_info["state_decoder"] is not None else None
            task_args.obs_decoder = dill.loads(env_info["obs_decoder"]) if env_info["obs_decoder"] is not None else None

            # Default/Base scheme
            scheme = {
                "state": {"vshape": env_info["state_shape"]},
                "obs": {"vshape": env_info["obs_shape"], "group": "agents",
                        "vshape_decoded": env_info.get("obs_shape_decoded", env_info["obs_shape"])},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            groups = {
                "agents": task_args.n_agents
            }
            preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
            }

            buffer = ReplayBuffer(scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
                                  preprocess=preprocess,
                                  device="cpu" if task_args.buffer_cpu_only else args.device)
            buffers[task_name] = buffer

            # Setup multiagent controller here
            mac = mac_REGISTRY[task_args.mac](buffer.scheme, groups, task_args)

            #point model to same object
            macs[task_name] = mac
            mac.agent = macs[task_names[0]].agent

            # Give runner the scheme
            runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

            # Learner
            learner = le_REGISTRY[task_args.learner](mac, buffer.scheme, logger, task_args)
            learners[task_name] = learner

            if task_args.use_cuda:
                learner.cuda()

            #if agent_state_dict is None:
            #    agent_state_dict = mac.agent.state_dict()
            # else:
            #    # copy all weights that have same dimensions
            #    sd = mac.agent.state_dict()
            #    for k, v in agent_state_dict.items():
            #        if (k in sd) and (v.shape == sd[k].shape):
            #            setattr(mac.agent, k, v)


            if task_args.checkpoint_path != "":

                timesteps = []
                timestep_to_load = 0

                if not os.path.isdir(task_args.checkpoint_path):
                    logger.console_logger.info("Checkpoint directory {} doesn't exist".format(task_args.checkpoint_path))
                    return

                # Go through all files in args.checkpoint_path
                for name in os.listdir(task_args.checkpoint_path):
                    full_name = os.path.join(task_args.checkpoint_path, name)
                    # Check if they are dirs the names of which are numbers
                    if os.path.isdir(full_name) and name.isdigit():
                        timesteps.append(int(name))

                if task_args.load_step == 0:
                    # choose the max timestep
                    timestep_to_load = max(timesteps)
                else:
                    # choose the timestep closest to load_step
                    timestep_to_load = min(timesteps, key=lambda x: abs(x - task_args.load_step))

                model_path = os.path.join(task_args.checkpoint_path, str(timestep_to_load))

                logger.console_logger.info("Loading model from {}".format(model_path))
                learner.load_models(model_path)
                runner.t_env = timestep_to_load

                if task_args.evaluate or task_args.save_replay:
                    evaluate_sequential(task_args, runner)
                    return
        return


    from copy import deepcopy
    # agent_state_dict = setup_components(logger, agent_state_dict)
    setup_components(logger, agent_state_dict)

    # start reptile training
    episode_ctrs = {k:0 for k, _ in sorted(task_configs.items())}
    last_test_Ts = {k:-v.test_interval - 1 for k, v in sorted(task_configs.items())}
    last_times = {k:time.time() for k, v in sorted(task_configs.items())}
    model_save_times = {k:0 for k, _ in sorted(task_configs.items())}
    start_time = time.time()

    logger.console_logger.info("Beginning REPTILE training ...")

    previous_task_id = None
    unfinished_tasks = {k for k, v in task_configs.items() if episode_ctrs[k] <= v.t_max}
    while len(unfinished_tasks):
        # INNER LOOP
        unfinished_tasks = {k for k, v in task_configs.items() if episode_ctrs[k] <=v.t_max}

        # pick task
        from random import randint
        task_id = sorted(list(unfinished_tasks))[randint(0, len(unfinished_tasks)-1)]

        logger.console_logger.info("Chose task {} at global counter {}".format(task_id, sum(episode_ctrs.values())))

        # roll out task a couple of times
        for t in range(args.n_task_rollouts[task_id]):
            episode_batch = runners[task_id].run(test_mode=False)
            buffers[task_id].insert_episode_batch(episode_batch)
            # train on task
            episode_ctrs[task_id] += 1
            if episode_ctrs[task_id] >= task_configs[task_id].t_max:
                break

        # reset mac weights
        # copy all weights that have same dimensions from last chosen task (not sure whether this is not redundant)
        if previous_task_id is not None:
            sd = macs[task_id].agent.state_dict()
            for k, v in macs[previous_task_id].agent.state_dict().items():
                if (k in sd) and (v.shape == sd[k].shape):
                    setattr(macs[task_id].agent, k, v)

        # train
        for t in range(args.n_task_trains[task_id]):

            if buffers[task_id].can_sample(task_configs[task_id].batch_size):
                episode_sample = buffers[task_id].sample(task_configs[task_id].batch_size)
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]
                if episode_sample.device != task_configs[task_id].device:
                    episode_sample.to(task_configs[task_id].device)

                learners[task_id].train(episode_sample,
                                        runners[task_id].t_env,
                                        episode_ctrs[task_id])

        # update weights of same dimensions using simple rule (otherwise: formulate as a gradient procedure)
        import operator
        for _task_id, _ in sorted(task_configs.items()):
            mac_state_dict = macs[task_id].agent.state_dict()
            if _task_id != task_id:
                _mac_state_dict = macs[_task_id].agent.state_dict()
                for k, v in _mac_state_dict.items():
                    if (k in mac_state_dict) and (v.shape == mac_state_dict[k].shape):
                        new_weights = operator.attrgetter(k)(macs[_task_id].agent) + args.reptile_epsilon * (mac_state_dict[k] - v)
                        setattr(macs[_task_id].agent, k, new_weights)
                        # agent_state_dict[k] += args.reptile_epsilon * (mac_state_dict[k] - macs[_task_id].agent.state_dict()[k])


        for task_id, task_config in task_configs.items():
            # Execute test runs once in a while
            n_test_runs = max(1, task_configs[task_id].test_nepisode // runners[task_id].batch_size)
            if (runners[task_id].t_env - last_test_Ts[task_id]) / task_configs[task_id].test_interval >= 1.0:
                loggers[task_id].console_logger.info("Now testing: {}".format(task_id))
                loggers[task_id].console_logger.info("t_env: {} / {}".format(runners[task_id].t_env,
                                                                             task_configs[task_id].t_max))
                loggers[task_id].console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_times[task_id],
                              last_test_Ts[task_id],
                              runners[task_id].t_env,
                              task_configs[task_id].t_max),
                    time_str(time.time() - start_time)))
                last_times[task_id] = time.time()

                last_test_Ts[task_id] = runners[task_id].t_env
                for _ in range(n_test_runs):
                    runners[task_id].run(test_mode=True)

        previous_task_id = task_id

        for task_id, task_config in task_configs.items():
            if task_config.save_model and \
                    (runners[task_id].t_env - model_save_times[task_id] >= task_config.save_model_interval or
                     model_save_times[task_id] == 0):
                model_save_times[task_id] = runners[task_id].t_env
                save_path = os.path.join(task_config.local_results_path,
                                         "models",
                                         task_config.unique_token,
                                         str(runners[task_id].t_env))
                #"results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learners[task_id].save_models(save_path)