from typing import List,Any,Dict

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.tools.coding.python_executor import execute_code_get_return
from GDesigner.utils.log import agent_debug_print

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        for id, info in spatial_info.items():
            spatial_str += (
                f"Agent {id} as a {info['role']} his answer to this question is:\n\n"
                f"{info['output']}\n\n"
            )
        for id, info in temporal_info.items():
            temporal_str += (
                f"Agent {id} as a {info['role']} his answer to this question was:\n\n"
                f"{info['output']}\n\n"
            )
        if spatial_str:
            user_prompt += (
                "At the same time, there are the following responses to the same "
                f"question for your reference:\n\n{spatial_str} \n\n"
            )
        if temporal_str:
            user_prompt += (
                "In the last round of dialogue, there were the following responses "
                f"to the same question for your reference: \n\n{temporal_str}"
            )
        return system_prompt, user_prompt

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        agent_debug_print(f"#################system_prompt:{system_prompt}")
        agent_debug_print(f"#################user_prompt:{user_prompt}")
        agent_debug_print(f"#################response:{response}")
        return response
