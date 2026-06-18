"""
Off-policy replay buffer (Section 4.3: "Transitions are stored in an
off-policy replay buffer following standard SAC training").

Each transition stores everything needed to recompute losses later:
    state, next_state       : encoded user states
    a_tilde (behavior)       : pre-squash latent action that was actually used
    reward                   : scalar reward R_t (Eq. 9)
    slate_items, rel_scores  : needed to recompute F^sub_theta for L_sub
    candidate_items          : full candidate pool (for diversity-rank negatives)
    done                     : episode-termination flag

Example:
    buf = ReplayBuffer(capacity=10000)
    buf.push(state, next_state, a_tilde, reward, slate, rels, candidates, done)
    batch = buf.sample(32)
"""

import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", [
    "state", "next_state", "a_tilde", "reward",
    "slate_items", "rel_scores", "candidate_items", "done",
])


class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, next_state, a_tilde, reward,
            slate_items, rel_scores, candidate_items, done):
        self.buffer.append(Transition(
            state, next_state, a_tilde, reward,
            slate_items, rel_scores, candidate_items, done,
        ))

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
