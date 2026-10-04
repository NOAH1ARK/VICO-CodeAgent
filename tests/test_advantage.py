"""Tests for dual-dimension step rewards and step-level advantages (reward/advantage.py)."""
import math

from reward.advantage import step_level_advantages, step_reward, z_score_normalize


def test_step_reward_dual_dimension():
    assert step_reward({"gt_ratio": 1.0, "syn_ratio": 1.0}, 0.3, 2048) == 1.3
    assert step_reward({"gt_ratio": 0.5, "syn_ratio": 0.0}, 0.3, 2048) == 0.5


def test_truncated_generation_gets_zero():
    assert step_reward({"gt_ratio": 1.0, "syn_ratio": 1.0, "response_length": 2048}, 0.3, 2048) == 0.0


def test_z_score_constant_group_is_zero():
    assert z_score_normalize([0.7, 0.7, 0.7]) == [0.0, 0.0, 0.0]


def test_z_score_is_standardized():
    z = z_score_normalize([0.0, 0.5, 1.0, 1.3])
    assert abs(sum(z)) < 1e-9
    assert math.isclose(sum(x * x for x in z) / len(z), 1.0, rel_tol=1e-9)


def test_finished_trajectories_hold_last_score():
    # traj 0 solved at step 1 and stopped; traj 1 fixed its bug at step 2; traj 2 never improved
    R = [[1.0], [0.0, 1.0], [0.0, 0.0]]
    adv = {(i, s): a for i, s, a in step_level_advantages(R)}
    # step 0: [1, 0, 0] -> traj 0 positive, others negative
    assert adv[(0, 0)] > 0 and adv[(1, 0)] < 0 and adv[(2, 0)] < 0
    # step 1: held [1, 1, 0] -> the repair step of traj 1 is rewarded, no sample for finished traj 0
    assert adv[(1, 1)] > 0 and adv[(2, 1)] < 0
    assert (0, 1) not in adv


def test_zero_advantage_samples_are_dropped():
    assert step_level_advantages([[0.5, 0.5], [0.5, 0.5]]) == []


def test_empty_trajectories_are_ignored():
    out = step_level_advantages([[], [1.0], [0.0]])
    assert {i for i, _, _ in out} == {1, 2}
