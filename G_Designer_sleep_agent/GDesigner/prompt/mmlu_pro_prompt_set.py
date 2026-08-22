from GDesigner.prompt.mmlu_prompt_set import MMLUPromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

_ANSWER_FORMAT = """
I will ask you a multiple-choice question with options labeled from A up to J.
The exact number of options is shown in the question; only one offered option is
correct. Use other agents' reasoning as advice, but check it critically.
Keep your reply under 100 words and include a brief step-by-step analysis.
The first line of your reply must contain only the selected option letter.
Never output a letter whose option is absent from the question.
"""

@PromptSetRegistry.register("mmlu_pro")
class MMLUProPromptSet(MMLUPromptSet):

    @staticmethod
    def get_constraint(role=None):
        from GDesigner.prompt.mmlu_prompt_set import ROLE_DESCRIPTION

        if role and role in ROLE_DESCRIPTION:
            return ROLE_DESCRIPTION[role].strip() + "\n\n" + _ANSWER_FORMAT
        return _ANSWER_FORMAT

    @staticmethod
    def get_analyze_constraint(role):
        from GDesigner.prompt.mmlu_prompt_set import ROLE_DESCRIPTION

        role_description = ROLE_DESCRIPTION.get(role, "")
        return (
            f"{role_description}\n\n{_ANSWER_FORMAT}\n"
            "Update your answer after critically checking the other agents. "
            "Do not imitate their analysis blindly."
        )

    @staticmethod
    def get_decision_constraint():
        return """
Choose the single correct answer to the multiple-choice question from the options
actually present (labeled A through at most J). Consider the other agents'
answers and analysis critically. Reply with exactly one valid option letter and
no other characters.
"""

    def get_description(self, role):
        description = super().get_description(role)
        return description.replace(
            "multiple-choice (A–D)", "multiple-choice (A–J)"
        )
