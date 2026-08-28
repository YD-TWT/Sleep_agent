from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMLU_CODE = ROOT / "code" / "MMLU"
sys.path.insert(0, str(MMLU_CODE))

from LLMLP import LLMLP
from online_sleep_attack import apply_letter_perturb, shift_letter


ROLES = [
    "Omniscient Domain Expert",
    "Omniscient Scientific Expert",
    "Economist",
    "Doctor",
    "Lawyer",
    "Mathematician",
]


class OnlineSleepAttackTest(unittest.TestCase):
    def test_letter_rotation(self):
        self.assertEqual([shift_letter(letter) for letter in "ABCD"], list("BCDA"))
        reply, answer, event = apply_letter_perturb("Reasoning\n(A)", "A")
        self.assertEqual(answer, "B")
        self.assertIn("Answer: (B)", reply)
        self.assertEqual(event["clean_answer"], "A")

    def _build_mock_network(self):
        llmlp = LLMLP(
            "mock",
            agents=6,
            agent_roles=ROLES,
            rounds=3,
            sleep_agent_indices=[0, 1],
        )
        round_answers = ["A", "A", "A", "B", "B", "C"]

        for node_idx, node in enumerate(llmlp.nodes):
            agent_idx = node_idx % llmlp.agents

            def activate(_question, node=node, agent_idx=agent_idx):
                node.active = True
                node.reply = f"agent-{agent_idx}"
                node.answer = round_answers[agent_idx]
                node.prompt_tokens = 0
                node.completion_tokens = 0

            node.activate = activate

        def select_sleep_responses(responses, *_args):
            selected = [
                position
                for position, response in enumerate(responses)
                if response in {"agent-0", "agent-1"}
            ]
            return selected, 0, 0

        llmlp.activation = select_sleep_responses
        return llmlp

    def test_ranker_selected_agent_slots_are_activated(self):
        llmlp = self._build_mock_network()
        llmlp.forward("question", attack_mode=False)
        final_slots = {
            node_idx % llmlp.agents for node_idx in llmlp.final_agent_indices
        }
        self.assertEqual(final_slots, {0, 1})

    def test_selected_sleep_agents_attack_only_in_final_round(self):
        llmlp = self._build_mock_network()
        result, *_ = llmlp.forward("question", attack_mode=True)
        self.assertEqual(result, "B")
        self.assertEqual(len(llmlp.sleep_attack_events), 2)
        self.assertTrue(all(event["reference_answer"] == "A"
                            for event in llmlp.sleep_attack_events))
        attacked_nodes = [node for node in llmlp.nodes if node.attack_applied]
        self.assertEqual(
            {llmlp.nodes.index(node) // llmlp.agents for node in attacked_nodes},
            {2},
        )


if __name__ == "__main__":
    unittest.main()
