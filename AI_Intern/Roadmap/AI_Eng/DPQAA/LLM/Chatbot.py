import httpx

from langchain_openai import ChatOpenAI as ChatModel

from langchain.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.types import Send

from IPython.display import Image, display

from Retrieval.Storage import Storage, SearchResult
from LLM.RAG import RAG

from Query.QueryRoute import QueryRoute, QueryRouteFPT, QueryType
from Query.QueryResolver import ResolvedQuery
from Query.QueryParallelizer import QueryPlan

from Tools.CodeExecutor import Language
from Tools.WebSearch import WebSearch

from Graph.State import ChatState

class Chatbot:
    '''Responsible for genereating a RAG pipeline'''
    MAX_WEB_CONTEXT_CHARS = 10000
    EXECUTION_SERVICE_IP = "http://127.0.0.1:8000/execute"

    def __init__(self, db: Storage, llm: ChatModel):
        self.db = db
        self.llm = llm
        self.web_search = WebSearch()
        self.router = llm.with_structured_output(QueryRoute)
        self.query_resolver = llm.with_structured_output(ResolvedQuery)
        self.router_fpt = llm.with_structured_output(QueryRouteFPT)
        self.decomposer = llm.with_structured_output(QueryPlan)

        self.checkpointer = MemorySaver()
        self.graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(ChatState)

        graph.add_node("resolve", self.resolve_node)
        graph.add_node("decompose", self.decompose_node)
        graph.add_node("execute_task", self.execute_task)
        graph.add_node("prepare", self.prepare_node)
        graph.add_node("generate", self.generate_node)

        graph.add_edge(START, "resolve")
        graph.add_edge("resolve", "decompose")

        graph.add_conditional_edges(
            "decompose",
            self.fan_out
        )

        graph.add_edge("execute_task", "prepare")
        graph.add_edge("prepare", "generate")
        graph.add_edge("generate", END)

        return graph.compile(
            checkpointer=self.checkpointer
        )

    def resolve_node(self, state: ChatState):
        self.emit_status(
            stage="resolve",
            message="Resolving query..."
        )

        messages = state["messages"]

        conversation = "\n".join(
            f"{message.type}: {message.content}"
            for message in messages
        )

        latest_query = messages[-1].content

        prompt = f"""
            {RAG.history_resolver_prompt}

            CONVERSATION:
            {conversation}

            USER'S LATEST QUERY:
            {latest_query}    
        """

        resolved = self.query_resolver.invoke(prompt)

        return {
            "resolved_query": resolved.query,
            "previous_context": resolved.context
        }

    def decompose_node(self, state: ChatState):
        self.emit_status(
            stage="decompose",
            message="Breaking query into tasks..."
        )

        query = state["resolved_query"]

        prompt = f"""
            {RAG.query_parallelizer_prompt}

            USER'S QUERY:
            {query}
        """

        plan = self.decomposer.invoke(prompt)
        tasks = plan.tasks

        if not tasks:
            tasks = [query]

        return {
            "tasks": tasks
        }

    def prepare_node(self, state: ChatState):
        self.emit_status(
            stage="prepare",
            message="Preparing response..."
        )

        task_sections = []

        for index, result in enumerate(state.get("task_results", []), start=1):
            task_sections.append(
                f"""
                    TASK {index}

                    USER'S TASK:
                    {result["task"]}

                    ROUTE:
                    {result["route"]}

                    INSTRUCTION:
                    {result["instruction"]}

                    RETRIEVED DATA:
                    {result["context"]}
                """
            )

        generation_context = "\n".join(task_sections)
        generation_context += f"""
            PREVIOUS CONVERSATION CONTEXT:
            {state.get("previous_context", "")}
        """

        return {
            "generation_context": generation_context
        }

    def generate_node(self, state: ChatState):
        self.emit_status(
            stage="generate",
            message="Generating response..."
        )

        system_message = SystemMessage(
            content=f"""
                You are an internal FPT Software policy assistant.

                You must follow these rules:
                {RAG.rules}

                Stylish your answers using Markdown following these rules:
                {RAG.markdown_rules}

                Stylish Math expression following these rules:
                {RAG.math_rules}

                Cite the sources following these rules:
                {RAG.cite_rules}
            """
        )

        user_message = HumanMessage(
            content = f"""
                {state["generation_context"]}

                USER QUESTION:
                {state["resolved_query"]}
            """
        )

        messages = [
            system_message,
            *state["messages"],
            user_message
        ]

        writer = get_stream_writer()
        full_response = []

        for chunk in self.llm.stream(messages):
            if not chunk.content:
                continue

            content = chunk.content
            full_response.append(content)

            writer({
                "type": "content",
                "content": content
            })

        response = "".join(full_response)

        writer({
            "type": "done"
        })

        return {
            "messages": [
                AIMessage(content=response)
            ],
            "response": response
        }

    def stream(self, query: str, user_id: str, session_id: str):
        state = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "session_id": session_id
        }

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        for event in self.graph.stream(
            state,
            config=config,
            stream_mode="custom"
        ):
            yield event

    def emit_status(self, stage: str, message: str):
        writer = get_stream_writer()

        writer({
            "type": "status",
            "stage": stage,
            "message": message
        })
        
    def build_context(self, results: list[SearchResult], max_chars_per_chunk: int=10000) -> str:
        '''Build full context based on the retrieved answer'''
        context_parts = []
        root_index, sub_index = 0, 0
        groups = self.db.group_by_root(results=results)

        for root_id, chunks in groups.items():
            root = self.db.get_chunk(chunk_id=root_id)
            root_payload = root.payload
            source = f"""
                FULL CONTEXT {root_index}
                Document: {root_payload["document_name"]}
                Section: {root_payload["title"]}
                Page(s): {root_payload["pages"]}

                {root_payload["full_content"]}
            """

            root_index += 1
            sub_index = 0
            current_size = len(source)
            for chunk in chunks:
                payload = chunk.payload
                sub =  f"""
                    SUB-CONTEXT {sub_index}
                    Document: {payload["document_name"]}
                    Section: {payload["title"]}
                    Page(s): {payload["pages"]}

                    {payload["content"]}\n
                """
                if current_size + len(sub) > max_chars_per_chunk:
                    break

                current_size += len(sub)
                source += sub
                sub_index += 1

            context_parts.append(source)

        return "\n".join(context_parts)

    def execute_python(self, code: str):
        '''Execute a python code and return the result'''
        response = httpx.post(
            self.EXECUTION_SERVICE_IP,
            json={
                "language": Language.PYTHON.value,
                "code": code
            },
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    def draw_graph(self, file: str | None):
        png = Image(self.graph.get_graph().draw_mermaid_png())

        if file is None:
            display(png)
            return

        with open(file, "wb") as f:
            f.write(png)

    def fan_out(self, state: ChatState):
        return [
            Send(
                "execute_task",
                {
                    "task": task,
                    "user_id": state["user_id"],
                    "session_id": state["session_id"],
                    "previous_context": state.get("previous_context", "")
                }
            )
            for task in state["tasks"]
        ]

    def execute_task(self, state: ChatState):
        task = state["task"]

        self.emit_status(
            stage="task_route",
            message=f"Routing task: {task}"
        )

        route = self.router.invoke(
            f"""
                {RAG.router_prompt}

                USER'S TASK:
                {task}
            """
        )

        match route.query_type:
            case QueryType.FPT:
                result = self.execute_fpt(task)
            case QueryType.COMPUTE:
                result = self.execute_compute(task)
            case QueryType.CODE:
                result = self.execute_code(task)
            case _:
                 result = self.execute_direct(task)

        return {
            "task_results": [
                {
                    "task": task,
                    "route": route.query_type.value,
                    "context": result.get("context", ""),
                    "instruction": result.get("instruction", "")
                }
            ]
        }

    def execute_fpt(self, task: str):
        self.emit_status(
            stage="document_search",
            message=f"Searching FPT documents: {task}"
        )

        results = self.db.search(
            query=task,
            limit=5
        )

        if not results:
            return {
                "context": "",
                "instruction": RAG.fpt_internal_not_found_instruction
            }

        context = self.build_context(results)

        # Check whether the retrieved information is sufficient
        prompt = f"""
            {RAG.router_fpt_prompt}

            Query:
            {task}

            SOURCES:
            {context}
        """

        self.emit_status(
            stage="evaluate",
            message="Evaluating..."
        )

        evaluation = self.router_fpt.invoke(prompt)

        if evaluation.lack_context:
            self.emit_status(
                stage="web_search",
                message=f"Searching web for: {task}"
            )

            context = self.web_search.search(
                query=task,
                max_results=5,
                search_depth="advanced"
            )

            if len(context) > self.MAX_WEB_CONTEXT_CHARS:
                context = (
                    context[:self.MAX_WEB_CONTEXT_CHARS]
                    + "\n[Web search results truncated]"
                )

            return {
                "context": context,
                "instruction": RAG.fpt_internal_not_found_instruction
            }

        return {
            "context": context,
            "instruction": ""
        }

    def execute_compute(self, task: str):
        self.emit_status(
            stage="compute",
            message=f"Calculating: {task}"
        )

        prompt = f"""
            {RAG.code_solver_instruction}

            USER'S QUERY:
            {task}
        """

        code = self.llm.invoke(prompt).content

        execution = self.execute_python(code)

        if execution["exit_code"] != 0:
            return {
                "context": RAG.rejection,
                "instruction": (
                    "Explain that the computation could not be completed."
                )
            }

        return {
            "context": (
                f"COMPUTATION RESULT:\n"
                f"{execution['stdout']}"
            ),
            "instruction": (
                "Answer the computational question using only "
                "the provided computed result."
            )
        }

    def execute_code(self, task: str):
        return {
            "context": "",
            "instruction": RAG.code_instruction
        }

    def execute_direct(self, task: str):
        route = self.router.invoke(
            f"""
            {RAG.router_prompt}

            USER'S TASK:
            {task}
            """
        )

        instruction = ""

        match route.query_type:
            case QueryType.CHIT_CHAT:
                instruction = RAG.chitchat_instruction

            case QueryType.GREETING:
                instruction = RAG.greeting_instruction

            case QueryType.RUBBISH:
                instruction = RAG.rubbish_instruction

            case QueryType.HARMFUL:
                instruction = RAG.harmful_rejection

            case QueryType.OTHER:
                instruction = RAG.other_instruction

        return {
            "context": "",
            "instruction": instruction
        }