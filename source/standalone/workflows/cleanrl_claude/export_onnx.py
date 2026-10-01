"""Export a cleanrl_claude checkpoint's actor and critic to a plain ONNX graph.

No Isaac Sim, no env: every dimension is read from the checkpoint's own weight shapes and saved args, so this
runs with the Isaac Lab python directly, no `pysaac`/AppLauncher needed:

    /home/GTL/fsangare/isaaclab/bin/python source/standalone/workflows/cleanrl_claude/export_onnx.py \
        --checkpoint logs/cleanrl_claude/ppo/Isaac-KingfisherSail-Direct-v0/<run>/model_last.pt

Output graph, per algorithm family (obs is the only input unless noted):

  ppo, ppo_rnd, ppo_rnd_rnn      (action, log_prob, value[, value_int])
  ppo_rnn                        (action, log_prob, value, h_out, c_out)      inputs: obs, h_in, c_in, reset
  ppo_discrete                   (action, log_prob, value)                    action: one integer level per action dim
  sac                            (action, log_prob, value)                    value needs a saved critic (see below)
  ddpg                           (action, value)                              no log_prob: the policy is deterministic
  dqn                            (action, q_values, value)                    value = max_a Q(obs, a); no actor/log_prob
  drqn                           (action, q_values, value, h_out, c_out)      inputs: obs, h_in, c_in

`action` is the policy's own deterministic choice (mean / mode / tanh(mean) / argmax), matching what eval.py runs
with `--stochastic` left off. `value` is V(obs) for the PPO family, Q(obs, action) for SAC/DDPG (the critic
conditioned on the actor's own action), max_a Q(obs, a) for DQN/DRQN.

IMPORTANT about log_prob for ppo / ppo_rnn / ppo_rnd / ppo_rnd_rnn: their action distribution is Normal(mu(obs),
std) with a STATE-INDEPENDENT std (one Parameter, not a function of obs). The log-density of the action actually
exported (the mean itself, since action == mu) is then just -sum(log std) - 0.5 * d * log(2*pi): a CONSTANT, the
same number for every observation. It is exported because it is well-defined and it is what "log-prob of the
actor's action" means for a deterministic mean-action policy here, but it carries no information about the
state. ppo_discrete (Categorical) and sac (state-dependent std) don't have this issue: their log_prob does vary
with obs.

SAC/DDPG critics: only checkpoints saved after this feature was added carry critic weights ("critic" key in the
.pt file, added alongside the actor's). Older checkpoints have no critic in them and are exported actor-only,
with a warning -- retrain, or resume is not supported, to get an exportable critic.

    /home/GTL/fsangare/isaaclab/bin/python source/standalone/workflows/cleanrl_claude/export_onnx.py \
        --checkpoint <run>/model_last.pt --output my_policy.onnx --batch_size 1 --skip_verify
"""
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro

import common


@dataclass
class Args:
    checkpoint: str  # a model_*.pt written by any cleanrl_claude training script
    output: str = ""  # default: <checkpoint dir>/<algo>.onnx
    opset: int = 17
    batch_size: int = 0  # 0 = dynamic batch dimension; a fixed number bakes it into the graph
    skip_verify: bool = False  # skip the onnxruntime cross-check against the torch output


LOG_2PI = math.log(2 * math.pi)


# ----------------------------------------------------------------------------- export wrappers
# Each wrapper's forward() uses only plain tensor ops (no torch.distributions, no Python branching on tensor
# values), which is what makes it traceable to ONNX. Reusing the training Agent/Actor classes where their
# forward() already returns raw tensors (ppo / ppo_rnd / ppo_rnn); reimplementing the few spots that don't
# (ppo_discrete's Categorical, sac's tanh-Normal) directly with the same maths as the training script.

class MlpGaussianExport(nn.Module):
    """ppo, ppo_rnd, ppo_rnd_rnn: Agent(obs) -> (mu, value[, value_int]), state-independent logstd."""

    def __init__(self, agent, clip_actions, obs_mean=None, obs_var=None):
        super().__init__()
        self.agent, self.clip_actions = agent, clip_actions
        self.normalize = obs_mean is not None
        if self.normalize:
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_var", obs_var)

    def forward(self, obs):
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8) if self.normalize else obs
        x = x.clamp(-10.0, 10.0) if self.normalize else x
        out = self.agent(x)
        mu, values = out[0], out[1:]
        action = mu.clamp(-self.clip_actions, self.clip_actions)
        log_prob = torch.ones_like(mu[:, 0]) * (-self.agent.logstd - 0.5 * LOG_2PI).sum()  # see module docstring
        return (action, log_prob, *values)


class RecurrentGaussianExport(nn.Module):
    """ppo_rnn: single-step LSTM policy. `reset` (B,) = 1.0 zeroes the incoming state (new episode)."""

    def __init__(self, agent, clip_actions):
        super().__init__()
        self.agent, self.clip_actions = agent, clip_actions

    def forward(self, obs, h_in, c_in, reset):
        mu, value, (h_out, c_out) = self.agent(obs.unsqueeze(0), (h_in, c_in), reset.unsqueeze(0))
        action = mu[0].clamp(-self.clip_actions, self.clip_actions)
        log_prob = torch.ones_like(action[:, 0]) * (-self.agent.logstd - 0.5 * LOG_2PI).sum()
        return action, log_prob, value[0], h_out, c_out


class DiscreteExport(nn.Module):
    """ppo_discrete: one categorical per action dimension, action = argmax (mode), log_prob = its log-density."""

    def __init__(self, agent, bins, obs_mean=None, obs_var=None):
        super().__init__()
        self.agent = agent
        self.register_buffer("bins", torch.tensor(bins))
        self.normalize = obs_mean is not None
        if self.normalize:
            self.register_buffer("obs_mean", obs_mean)
            self.register_buffer("obs_var", obs_var)

    def forward(self, obs):
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8) if self.normalize else obs
        x = x.clamp(-10.0, 10.0) if self.normalize else x
        h = self.agent.torso(x)
        logits = self.agent.logits(h).view(-1, self.agent.act_dim, self.agent.maxb)
        valid = torch.arange(self.agent.maxb, device=obs.device) < self.bins.unsqueeze(-1)
        log_p = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        action = log_p.argmax(-1)  # (B, act_dim): level index per action dimension, not a continuous value
        log_prob = log_p.max(-1).values.sum(-1)
        value = self.agent.value(h).squeeze(-1)
        return action, log_prob, value


class SACExport(nn.Module):
    """sac: deterministic action = tanh(mu); log_prob is the tanh-Normal density at that point (state-dependent std).
    value = Q0(obs, action) if a critic was saved in the checkpoint, else omitted (see the module docstring)."""

    def __init__(self, actor, q0=None):
        super().__init__()
        self.actor, self.q0 = actor, q0

    def forward(self, obs):
        mu, logstd = self.actor(obs)
        action = torch.tanh(mu)
        log_prob = (-logstd - 0.5 * LOG_2PI - torch.log(1 - action.pow(2) + 1e-6)).sum(-1)
        if self.q0 is None:
            return action, log_prob
        value = self.q0(torch.cat([obs, action], -1)).squeeze(-1)
        return action, log_prob, value


class DDPGExport(nn.Module):
    """ddpg: deterministic policy, no distribution -> no log_prob. value = Q(obs, action) if a critic was saved."""

    def __init__(self, actor, q=None):
        super().__init__()
        self.actor, self.q = actor, q

    def forward(self, obs):
        action = self.actor(obs)
        if self.q is None:
            return (action,)
        value = self.q(torch.cat([obs, action], -1)).squeeze(-1)
        return action, value


class DQNExport(nn.Module):
    """dqn: no actor/log_prob (value-based). action = argmax Q, value = max Q (the value of the greedy policy)."""

    def __init__(self, q):
        super().__init__()
        self.q = q

    def forward(self, obs):
        q_values = self.q(obs)
        return q_values.argmax(-1), q_values, q_values.max(-1).values


class DRQNExport(nn.Module):
    """drqn: single LSTMCell step. h_in/c_in, h_out/c_out shape (B, hidden) -- NOT (1, B, hidden) like ppo_rnn's LSTM."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs, h_in, c_in):
        q_values, (h_out, c_out) = self.net.step(obs, (h_in, c_in))
        return q_values.argmax(-1), q_values, q_values.max(-1).values, h_out, c_out


# ----------------------------------------------------------------------------- build a wrapper from a checkpoint
def build(ck):
    """Returns (module, example_inputs, input_names, output_names, dynamic_axes) for torch.onnx.export."""
    import drqn, ppo, ppo_discrete, ppo_rnd, ppo_rnn, sac
    # ddpg.py and dqn.py build their actor/critic inline in main() (no reusable class): rebuilt below with common.mlp

    algo, args, state = ck["algo"], ck["args"], ck["state"]
    obs_mean, obs_var = ck["obs_rms"] if ("obs_rms" in ck and args.get("normalize_input")) else (None, None)
    B = 2  # example batch size for tracing; irrelevant once dynamic_axes makes it a free dimension

    if algo in ("ppo", "ppo_rnd", "ppo_rnd_rnn"):
        D, A = state["torso.0.weight"].shape[1], state["mu.weight"].shape[0]
        agent = (ppo if algo == "ppo" else ppo_rnd).Agent(D, A, tuple(args["hidden_units"]), args["activation"])
        agent.load_state_dict(state)
        wrapper = MlpGaussianExport(agent, args["clip_actions"], obs_mean, obs_var)
        out_names = ["action", "log_prob", "value"] + (["value_int"] if algo != "ppo" else [])
        return wrapper, (torch.randn(B, D),), ["obs"], out_names, {"obs": {0: "batch"}}

    if algo == "ppo_rnn":
        D, A, H = state["torso.0.weight"].shape[1], state["mu.weight"].shape[0], args["lstm_hidden"]
        agent = ppo_rnn.Agent(D, A, tuple(args["hidden_units"]), args["activation"], H)
        agent.load_state_dict(state)
        wrapper = RecurrentGaussianExport(agent, args["clip_actions"])
        ins = (torch.randn(B, D), torch.zeros(1, B, H), torch.zeros(1, B, H), torch.zeros(B))
        names = ["obs", "h_in", "c_in", "reset"]
        # h_in/c_in follow nn.LSTM's own state shape (num_layers=1, batch, hidden): batch is axis 1, not axis 0.
        axes = {"obs": {0: "batch"}, "h_in": {1: "batch"}, "c_in": {1: "batch"}, "reset": {0: "batch"}}
        return wrapper, ins, names, ["action", "log_prob", "value", "h_out", "c_out"], axes

    if algo == "ppo_discrete":
        bins = args["bins"]
        D = state["torso.0.weight"].shape[1]
        agent = ppo_discrete.Agent(D, len(bins), bins, tuple(args["hidden_units"]), args["activation"])
        agent.load_state_dict(state)
        wrapper = DiscreteExport(agent, bins, obs_mean, obs_var)
        return wrapper, (torch.randn(B, D),), ["obs"], ["action", "log_prob", "value"], {"obs": {0: "batch"}}

    if algo == "sac":
        D, A, H = state["net.0.weight"].shape[1], state["mu.weight"].shape[0], args["hidden"]
        actor = sac.Actor(D, A, H)
        actor.load_state_dict(state)
        q0 = None
        if "critic" in ck:
            q0 = common.mlp([D + A, H, H, 1], "relu")
            q0.load_state_dict({k[2:]: v for k, v in ck["critic"].items() if k.startswith("0.")})  # ModuleList[0]
        else:
            print("[export_onnx] no 'critic' in this checkpoint (trained before critic saving was added): "
                  "exporting the actor only, without a value output. Retrain to get an exportable critic.")
        wrapper = SACExport(actor, q0)
        names = ["action", "log_prob"] + (["value"] if q0 is not None else [])
        return wrapper, (torch.randn(B, D),), ["obs"], names, {"obs": {0: "batch"}}

    if algo == "ddpg":
        D, A, H = state["0.0.weight"].shape[1], state["0.4.weight"].shape[0], args["hidden"]
        actor = nn.Sequential(common.mlp([D, H, H, A], "relu"), nn.Tanh())
        actor.load_state_dict(state)
        q = None
        if "critic" in ck:
            q = common.mlp([D + A, H, H, 1], "relu")
            q.load_state_dict(ck["critic"])
        else:
            print("[export_onnx] no 'critic' in this checkpoint (trained before critic saving was added): "
                  "exporting the actor only, without a value output. Retrain to get an exportable critic.")
        wrapper = DDPGExport(actor, q)
        names = ["action"] + (["value"] if q is not None else [])
        return wrapper, (torch.randn(B, D),), ["obs"], names, {"obs": {0: "batch"}}

    if algo == "dqn":
        D, N_ACT, H = state["0.weight"].shape[1], state["4.weight"].shape[0], args["hidden"]
        q = common.mlp([D, H, H, N_ACT], "relu")
        q.load_state_dict(state)
        wrapper = DQNExport(q)
        return wrapper, (torch.randn(B, D),), ["obs"], ["action", "q_values", "value"], {"obs": {0: "batch"}}

    if algo == "drqn":
        D, N_ACT, H = state["enc.0.weight"].shape[1], state["out.weight"].shape[0], args["hidden"]
        net = drqn.RecurrentQNet(D, N_ACT, H)
        net.load_state_dict(state)
        wrapper = DRQNExport(net)
        ins = (torch.randn(B, D), torch.zeros(B, H), torch.zeros(B, H))
        names = ["obs", "h_in", "c_in"]
        return wrapper, ins, names, ["action", "q_values", "value", "h_out", "c_out"], {n: {0: "batch"} for n in names}

    raise ValueError(f"unknown algorithm '{algo}' in checkpoint")


# ----------------------------------------------------------------------------- verify + main
def verify(path, wrapper, example_inputs, input_names):
    """Cross-check the exported graph against the torch wrapper on a fresh random batch; prints the max abs diff."""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = wrapper(*example_inputs)
    feed = {n: t.numpy() for n, t in zip(input_names, example_inputs)}
    onnx_out = sess.run(None, feed)
    for name, t, o in zip(sess.get_outputs(), torch_out, onnx_out):
        diff = np.abs(t.numpy() - o).max()
        print(f"[export_onnx]   {name.name:10s} max|torch - onnx| = {diff:.3e}")


def main():
    args = tyro.cli(Args)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wrapper, example_inputs, input_names, output_names, dynamic_axes = build(ck)
    wrapper.eval()
    if args.batch_size:  # bake in a fixed batch size instead of a dynamic one (batch axis position varies: see build())
        resized = []
        for name, t in zip(input_names, example_inputs):
            batch_axis = next((ax for ax, label in dynamic_axes.get(name, {}).items() if label == "batch"), None)
            if batch_axis is None:
                resized.append(t)
            else:
                reps = [1] * t.dim()
                reps[batch_axis] = args.batch_size
                resized.append(t.narrow(batch_axis, 0, 1).repeat(reps).contiguous())
        example_inputs = tuple(resized)
        dynamic_axes = {k: {a: v for a, v in ax.items() if v != "batch"} for k, ax in dynamic_axes.items()}

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), f"{ck['algo']}.onnx")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    torch.onnx.export(wrapper, example_inputs, output, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes or None, opset_version=args.opset)
    print(f"[export_onnx] {ck['algo']}: wrote '{output}' "
          f"(inputs: {input_names}, outputs: {output_names})")

    if not args.skip_verify:
        verify(output, wrapper, example_inputs, input_names)


if __name__ == "__main__":
    main()
