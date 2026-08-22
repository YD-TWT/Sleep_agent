import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio

class Node(ABC):

    def __init__(self,
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="",
                 llm_name:str = "",
                 ):
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.decision_outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {
            'inputs': [],
            'outputs': [],
            'raw_inputs': [],
            'decision_outputs': [],
        }

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []

    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs
        self.last_memory['decision_outputs'] = list(getattr(self, "decision_outputs", []) or [])

    @staticmethod
    def _latest_text(bucket: Any) -> Any:
        if isinstance(bucket, list):
            return bucket[-1] if bucket else None
        return bucket

    def visible_output(self, for_decision: bool = False, from_memory: bool = False) -> Any:
        if from_memory:
            memory = self.last_memory
            if for_decision:
                decision = self._latest_text(memory.get("decision_outputs"))
                if decision not in (None, ""):
                    return decision
            return self._latest_text(memory.get("outputs"))
        if for_decision:
            decision = self._latest_text(getattr(self, "decision_outputs", None))
            if decision not in (None, ""):
                return decision
        return self._latest_text(self.outputs)

    def get_spatial_info(self, for_decision: bool = False)->Dict[str,Dict]:
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_output = predecessor.visible_output(for_decision=for_decision)
                if predecessor_output is None:
                    continue
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self, for_decision: bool = False)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_output = predecessor.visible_output(
                    for_decision=for_decision, from_memory=True
                )
                if predecessor_output is None:
                    continue
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return temporal_info

    def execute(self, input:Any, **kwargs):
        self.outputs = []
        self.decision_outputs = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    async def async_execute(self, input:Any, **kwargs):

        self.outputs = []
        self.decision_outputs = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ Process the raw_inputs(most of the time is a List[Dict]) """
