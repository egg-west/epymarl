import copy
import numpy as np

import ot
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from components.standarize_stream import RunningMeanStd
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.task_encoder_qmix import TaskEncoderQMixer
from modules.task_encoders.base_task_encoder import BaseTaskEncoder
from modules.dynamic_models.probabilistic_forward_model import ProbabilisticForwardModel
class AbstractQLearner:
    """Agent learner with state abstraction"""
    def __init__(self, mac, scheme, logger, args, wandb_logger):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.wandb_logger = wandb_logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                #self.mixer = QMixer(args)
                self.mixer = TaskEncoderQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)
        
        if args.use_task_encoder and args.independent_task_encoder:
            state_dim = int(np.prod(args.state_shape))
            joint_action_dim = int(args.n_actions * args.n_agents)
            self.forward_model = ProbabilisticForwardModel(
                args.state_embedding_dim,
                args.task_embedding_dim,
                [joint_action_dim],
                args.fm_hidden_dim,
            ).to(args.device)
            self.FM_optimiser = Adam(params=self.forward_model.parameters(), lr=args.lr)
            self.task_encoder = BaseTaskEncoder(self.args.num_embeddings, self.args.task_embedding_dim).to(args.device)
            self.TE_optimiser = Adam(params=self.task_encoder.parameters(), lr=args.lr)

            if args.use_agent_encoder:
                self.agent_forward_model = ProbabilisticForwardModel(
                    args.observation_embedding_dim,
                    args.task_embedding_dim + args.agent_embedding_dim,
                    [int(args.n_actions)],
                    256
                ).to(args.device)
                self.AFM_optimiser = Adam(params=self.agent_forward_model.parameters(), lr=args.lr)
                self.agent_encoder = BaseTaskEncoder(self.args.n_agents + 3, self.args.agent_embedding_dim).to(args.device)
                self.AE_optimiser = Adam(params=self.agent_encoder.parameters(), lr=args.lr)
                
                self.agent_indices = torch.arange(args.n_agents).to(args.device)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        current_task_id: int,
    ):
        bs = batch.batch_size
        seq_len = int(batch["task_indices_global"].shape[1] - 1)
        if self.args.use_task_encoder and self.args.independent_task_encoder:
            ####################### Update the forward model
            batch_task_indicies = batch["task_indices_global"][:, :-1]
            with torch.no_grad():
                task_embedding = self.task_encoder(batch_task_indicies).reshape([-1, self.args.task_embedding_dim])

            onehot_actions = F.one_hot(batch["actions"][:, :-1]).squeeze(-2)
            # batch["actions"].shape=torch.Size([32, 53, 8, 1])
            # onehot_actions.shape=torch.Size([32, 66, 8, 14])
            joint_actions = torch.cat([onehot_actions[:, :, i, :] for i in range(self.args.n_agents)], dim=-1)
            joint_actions = joint_actions.reshape((-1, self.args.n_actions * self.args.n_agents))
            
            h = self.mixer.state_embedding(batch["state"][:, :-1])
            # h.shape=torch.Size([2432, 64]), task_embedding.shape=torch.Size([2432, 64])
            # [1792, 112]
            fm_inputs = torch.cat([h.detach(), joint_actions, task_embedding], dim=1)
            #print(f'{fm_inputs.shape=}, {h.shape=}, {joint_actions.shape=}, {task_embedding.shape=}')
            #print(f'{self.args.state_shape=}')
            # fm_inputs.shape=torch.Size([2400, 304]), h.shape=torch.Size([2400, 128]), joint_actions.shape=torch.Size([2400, 112]), task_embedding.shape=torch.Size([2400, 64])
            # self.args.state_shape=56
            predicted_next_state, sigma = self.forward_model(fm_inputs)
            next_h = self.mixer.state_embedding(batch["state"][:, 1:])
            diff = (predicted_next_state - next_h.detach()) / sigma
            fm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

            self.FM_optimiser.zero_grad()
            fm_loss.backward()
            self.FM_optimiser.step()
            self.wandb_logger.log({"SA/fm_loss":fm_loss.item()}, t_env)

            ####################### Update the task encoder
            random_bias = torch.LongTensor(
                np.random.randint(low=1, high=self.args.n_train_tasks, size=batch_task_indicies.shape[0]) 
            ).to(self.args.device).unsqueeze(-1).tile((1, batch_task_indicies.shape[1])).unsqueeze(-1)
            #print(f'{random_bias.shape=}')
            #random_bias.shape=torch.Size([32]), batch["task_indices_global"].shape=torch.Size([32, 51, 1]), random_task_indicies.shape=torch.Size([32, 51, 32])

            random_task_indicies = (batch_task_indicies + random_bias) % self.args.n_train_tasks# every element of bias is in [1, n_tasks)
            # print(f'{batch["task_indices_global"].shape=}, {random_task_indicies.shape=}') # [32, 74, 1]
            
            random_task_embedding = self.task_encoder(random_task_indicies).reshape([-1, self.args.task_embedding_dim])
            with torch.no_grad():
                random_task_fm_inputs = torch.cat([h.detach(), joint_actions, random_task_embedding], dim=1)
                random_task_predicted_next_state, sigma = self.forward_model(random_task_fm_inputs,)

            next_state_diff = None
            if self.args.optimal_transport_loss:
                #print(f"{batch_task_indicies[:, 0, :].shape=}")
                # [bs, task_embedding_dim]
                task_embedding = self.task_encoder(batch_task_indicies[:, 0, :].squeeze(1))
                random_task_embedding = self.task_encoder(random_task_indicies[:, 0, :].squeeze(1))
                #print(f"{task_embedding.shape=}, {random_task_embedding.shape=}")
                task_embedding_unroll = task_embedding.unsqueeze(1).tile(1, seq_len).reshape(bs*seq_len, -1)
                # print(f"{task_embedding.shape=}") # task_embedding.shape=torch.Size([1088, 64])
                # error 1088x240 and 304x256

                fm_inputs = torch.cat([h.detach(), joint_actions, task_embedding_unroll], dim=1)
                predicted_next_state, sigma = self.forward_model(fm_inputs)
                diff = (predicted_next_state - next_h.detach()) / sigma
                fm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

                coupling_matrix_list = []
                cost_matrix = torch.sqrt(
                    torch.cdist(
                        predicted_next_state.reshape(bs, seq_len, -1),
                        random_task_predicted_next_state.reshape(bs, seq_len, -1),
                    )
                )
                ab = (torch.ones(seq_len) / seq_len).to(self.args.device)

                dis_list = []
                for i in range(bs):
                    M = cost_matrix[i]
                    #print(f"{cost_matrix.shape=}, {bs=}, {seq_len=}")
                    #print(f"{M.shape=}, {predicted_next_state[i*seq_len:(i+1)*seq_len].shape=}, {random_task_predicted_next_state[i*seq_len:(i+1)*seq_len].shape=}")
                    #loss = ot.emd2(ab, ab, M)
                    
                    gamma = ot.emd(ab, ab, M)
                    dis = (M * gamma).mean()
                    dis_list.append(dis.detach())

                te_diff = torch.norm(task_embedding - random_task_embedding, dim=1)
                next_state_diff = torch.stack(dis_list)
                te_loss = F.mse_loss(te_diff, next_state_diff)
                te_loss += fm_loss
                #print(f"{te_diff.shape=}, {next_state_diff.shape=}")
            else:
                # [2368, 64]
                task_embedding = self.task_encoder(batch_task_indicies).reshape([-1, self.args.task_embedding_dim])
                
                # te_diff = torch.norm(task_embedding - random_task_embedding)
                # next_state_diff = torch.norm(predicted_next_state.detach() - random_task_predicted_next_state)
                fm_inputs = torch.cat([h.detach(), joint_actions, task_embedding], dim=1)
                te_diff = torch.norm(task_embedding - random_task_embedding, dim=1)

                predicted_next_state, sigma = self.forward_model(fm_inputs)
                next_state_diff = torch.norm(predicted_next_state.detach() - random_task_predicted_next_state, dim=1)
                diff = (predicted_next_state - next_h.detach()) / sigma
                fm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
                te_loss = F.mse_loss(te_diff, next_state_diff)
                te_loss += fm_loss
            self.TE_optimiser.zero_grad()
            te_loss.backward()
            self.TE_optimiser.step()
            self.wandb_logger.log({"SA/te_loss":te_loss.item()}, t_env)

            if self.args.use_agent_encoder:
                with torch.no_grad():
                    task_embedding = self.task_encoder(batch["task_indices"][:, :-1]).squeeze(-2).reshape([bs*seq_len, self.args.n_agents, -1])
                #print(f"for agent encoder: {task_embedding.shape=}")
                agent_indices = self.agent_indices.unsqueeze(0).unsqueeze(0).tile([bs, seq_len, 1])
                #print(f"{agent_indices.shape=}") # expect [32, seq_len, 8] # [32, 68, 8]
                agent_embedding = self.agent_encoder(agent_indices).reshape(-1, self.args.n_agents, self.args.agent_embedding_dim)
                #print(f"{agent_embedding.shape=}") # expect [32 * seq_len, 8, 64] # [2176, 8, 64]
                onehot_actions = F.one_hot(batch["actions"][:, :-1]).squeeze(-2).to(batch.device) # [32, 66, 8, 14]
                agent_action = onehot_actions.reshape((-1, self.args.n_agents, self.args.n_actions)) # [2176, 8, 14]
                h = self.mac.observation_encode(batch["obs"][:, :-1]).reshape(bs*seq_len, self.args.n_agents, self.args.observation_embedding_dim)
                #print(f"{h.shape=}") # [32*68, 8, 64]
                afm_inputs = torch.cat([task_embedding, agent_embedding.detach(), h, agent_action], dim=2)
                #print(f"{afm_inputs.shape=}") # 2176, 8, 206
                predicted_next_observations, sigma = self.agent_forward_model(afm_inputs.detach())
                # print(f"{predicted_next_observations.shape=}") # [2176, 8, 64]
                next_h = self.mac.observation_encode(batch["obs"][:, 1:]).reshape(bs*seq_len, self.args.n_agents, self.args.observation_embedding_dim)
                diff = (predicted_next_observations - next_h.detach()) / sigma
                afm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

                self.AFM_optimiser.zero_grad()
                afm_loss.backward()
                self.AFM_optimiser.step()
                self.wandb_logger.log({"SA/afm_loss":afm_loss.item()}, t_env)

                #################################### update the agent embedding
                TEST_SEQ_LEN = 10
                TEST_SAMPLE_NUMBER = 10
                random_agent_indices_bias = torch.LongTensor(np.random.randint(low=1, high=self.args.n_agents, size=self.args.n_agents)).to(batch.device)
                # print(f"{self.agent_indices.shape=}, {random_agent_indices_bias.shape=}")# [8]
                random_agent_indices = (self.agent_indices + random_agent_indices_bias).unsqueeze(0).unsqueeze(0).tile([bs, seq_len, 1]) % self.args.n_agents
                #print(f"{random_agent_indices.shape=}") [32, 68, 8]
                random_agent_embedding = self.agent_encoder(random_agent_indices).reshape(-1, self.args.n_agents, self.args.agent_embedding_dim)

                #print(f"{random_agent_embedding.shape=}") # [2176, 8, 64]
                random_afm_inputs = torch.cat([task_embedding, random_agent_embedding, h.detach(), agent_action], dim=2)#[:TEST_SAMPLE_NUMBER]
                #print(f"{random_afm_inputs.shape=}") # [2176, 8, 206]
                random_agent_predicted_next_observation, sigma = self.agent_forward_model(random_afm_inputs)
                #print(f"{random_agent_predicted_next_observation.shape=}") # [2176, 8, 64]
                afm_inputs = torch.cat([task_embedding, agent_embedding, h.detach(), agent_action], dim=2)#[:TEST_SAMPLE_NUMBER]
                new_predicted_next_observations, sigma = self.agent_forward_model(afm_inputs)
                diff = (new_predicted_next_observations - next_h.detach()) / sigma
                afm_loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))

                # ae_diff = torch.norm(agent_embedding.reshape(-1, agent_embedding.shape[-1])[:TEST_SAMPLE_NUMBER] - random_agent_embedding.reshape(-1, random_agent_embedding.shape[-1])[:TEST_SAMPLE_NUMBER], dim=1)
                # next_observation_diff = torch.norm(
                #     new_predicted_next_observations.reshape(-1, new_predicted_next_observations.shape[-1])[:TEST_SAMPLE_NUMBER] - random_agent_predicted_next_observation.reshape(-1, random_agent_predicted_next_observation.shape[-1])[:TEST_SAMPLE_NUMBER],
                #     dim=1
                # )
                ae_diff = torch.norm(agent_embedding.reshape(-1, agent_embedding.shape[-1]) - random_agent_embedding.reshape(-1, random_agent_embedding.shape[-1]), dim=1)
                next_observation_diff = torch.norm(
                    new_predicted_next_observations.reshape(-1, new_predicted_next_observations.shape[-1]) - random_agent_predicted_next_observation.reshape(-1, random_agent_predicted_next_observation.shape[-1]),
                    dim=1
                )
                self.AE_optimiser.zero_grad()
                #ae_loss = F.mse_loss(ae_diff, next_observation_diff.detach()) + afm_loss
                ae_loss = F.mse_loss(ae_diff, next_observation_diff.detach())
                ae_loss.backward()
                self.AE_optimiser.step()
                self.wandb_logger.log({"SA/ae_loss":ae_loss.item()}, t_env)
                self.wandb_logger.log({"SA/afm_diff":torch.mean(diff).detach().item()}, t_env)

        ####################### Update the agent

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        task_indices = batch["task_indices"]#[:, :-1] # include many tasks

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        if self.args.independent_task_encoder:
            with torch.no_grad():
                task_embedding = self.task_encoder(task_indices).squeeze(-2)
            if self.args.use_agent_encoder:
                agent_embedding = self.agent_encoder(self.agent_indices).unsqueeze(0).tile([bs, 1, 1])
                # print(f"{agent_embedding.shape}")# [8, 64]
                #print(f"{agent_embedding.shape=}") # torch.Size([32, 8, 64])
                for t in range(batch.max_seq_length):
                    agent_outs = self.mac.forward(batch, t=t, task_embedding=task_embedding, agent_embedding=agent_embedding)
                    mac_out.append(agent_outs)
            else:
                for t in range(batch.max_seq_length):
                    agent_outs = self.mac.forward(batch, t=t, task_embedding=task_embedding)
                    mac_out.append(agent_outs)
        else:
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        if self.args.independent_task_encoder:
            if self.args.use_agent_encoder:
                agent_embedding = self.agent_encoder(self.agent_indices).unsqueeze(0).tile([bs, 1, 1])
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t, task_embedding=task_embedding, agent_embedding=agent_embedding)
                    target_mac_out.append(target_agent_outs)
            else:
                for t in range(batch.max_seq_length):
                    target_agent_outs = self.target_mac.forward(batch, t=t, task_embedding=task_embedding)
                    target_mac_out.append(target_agent_outs)
        else:
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            if self.args.use_task_encoder:
                if self.args.independent_task_encoder:
                    with torch.no_grad():
                        task_embed = self.task_encoder(batch["task_indices_global"][:, :-1]).reshape((-1, self.args.task_embedding_dim))
                        task_embed_next = self.task_encoder(batch["task_indices_global"][:, 1:]).reshape((-1, self.args.task_embedding_dim))
                else:
                    with torch.no_grad():
                        task_embed = self.target_mac.task_encode(batch["task_indices_global"][:, :-1])
                        task_embed_next = self.target_mac.task_encode(batch["task_indices_global"][:, 1:])
                        #print(f'{batch["state"].shape=}, {batch["state"][:, :-1].shape}')
                        #print(f'{batch["task_indices_global"].shape=}, {batch["task_indices_global"][:, :-1].shape}')
                        # batch["state"].shape=torch.Size([32, 70, 65]), torch.Size([32, 69, 65])
                        # batch["task_indices_global"].shape=torch.Size([32, 70, 1]), torch.Size([32, 69, 1])
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], task_embed)
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], task_embed_next)
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            train_stats = {}
            train_stats["train/loss"] = loss.item()
            train_stats["train/grad_norm"] = grad_norm.item()
            train_stats["train/td_error_abs"] = masked_td_error.abs().sum().item() / mask_elems
            train_stats["train/q_taken_mean"] = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
            train_stats["train/target_mean"] = (targets * mask).sum().item()/(mask_elems * self.args.n_agents)
            self.wandb_logger.log(train_stats, t_env)

            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            return train_stats

    def get_task_embedding(self, task_id):
        task_id = torch.LongTensor([task_id]).to(self.args.device)
        return self.task_encoder(task_id)
    
    def get_agent_embedding(self):
        return self.agent_encoder(self.agent_indices)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
