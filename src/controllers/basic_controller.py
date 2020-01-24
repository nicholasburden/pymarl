from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
from collections import OrderedDict



# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = th.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return th.index_select(a, dim, order_index.to(self.args.device))

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.action_input_representation == "Grid":
            obs = ep_batch["obs"][:, t]
            avail_actions = ep_batch["avail_actions"][:, t]
            avail_actions_grid = self.args.avail_actions_encoder(avail_actions, obs)
            agent_inputs["actions_2d"] = avail_actions_grid.view(-1, *avail_actions_grid.shape[-3:])
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
       
	    #print(sum(p.numel() for p in self.agent.parameters()))
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.action_input_representation=="InputFlat" or self.args.action_input_representation=="Grid":
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, self.args.n_actions, -1)  # bav
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = OrderedDict()
        obs = batch["obs"][:, t]
        if self.args.obs_decoder is not None:
            obs_decoded = self.args.obs_decoder(obs)
            inputs["2d"] = obs_decoded["2d"][0]
        else:
            inputs["1d"] = obs
        """
        if len(obs.shape[3:]) in [2, 3]:
            inputs["2d"] = obs_decoded["2d"]
        else:
            inputs["1d"] = obs
        """
        if self.args.obs_last_action:
            if t == 0:
                onehot = th.zeros_like(batch["actions_onehot"][:, t])
            else:
                onehot = batch["actions_onehot"][:, t - 1]
            inputs.update([("1d", th.cat([inputs["1d"], onehot], -1) if "1d" in inputs else onehot)])

        if self.args.obs_agent_id:
            obs_agent_id = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            inputs.update([("1d", th.cat([inputs["1d"], obs_agent_id], -1) if "1d" in inputs else obs_agent_id)])
        if self.args.action_input_representation == "InputFlat":
            avail_actions = th.eye(self.args.n_actions,self.args.n_actions).repeat(bs*self.n_agents,1)
            inputs["1d"] = inputs["1d"].reshape(bs*self.n_agents, -1)
            inputs["1d"] = self.tile(inputs["1d"], 0, self.args.n_actions)
            inputs["1d"] = th.cat((inputs["1d"].to(self.args.device), avail_actions.to(self.args.device)), -1)
            if self.args.obs_decoder is not None:
                inputs["2d"] = inputs["2d"].reshape(bs * self.n_agents, *inputs["2d"].shape[2:])
                inputs["2d"] = self.tile(inputs["2d"], 0, self.args.n_actions)
            return inputs

        inputs = OrderedDict([(k, v.reshape(bs * self.n_agents, *v.shape[2:])) for k, v in inputs.items()])
        if self.args.action_input_representation == "Grid":
            inputs["2d"] = self.tile(inputs["2d"], 0, self.args.n_actions)
            inputs["1d"] = self.tile(inputs["1d"], 0, self.args.n_actions)
        return inputs

    def _get_input_shape(self, scheme):

        if isinstance(scheme["obs"]["vshape"], int):
            scheme["obs"]["vshape"] = (scheme["obs"]["vshape"],)
        if isinstance(scheme["actions_onehot"]["vshape"], int):
            scheme["actions_onehot"]["vshape"] = (scheme["actions_onehot"]["vshape"],)
        if isinstance(scheme["obs"]["vshape_decoded"], int):
            scheme["obs"]["vshape_decoded"] = (scheme["obs"]["vshape_decoded"],)
        input_shape = OrderedDict()
        if len(scheme["obs"]["vshape_decoded"]) in [2,3]:  # i.e. multi-channel image data
            input_shape["2d"] = scheme["obs"]["vshape_decoded"]
        else:
            input_shape["1d"] = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape.update([("1d", (input_shape.get("1d", [0])[0] + scheme["actions_onehot"]["vshape"][0],))])
        if self.args.obs_agent_id:
            input_shape.update([("1d", (input_shape.get("1d", [0])[0] + self.n_agents,))])
        return input_shape
