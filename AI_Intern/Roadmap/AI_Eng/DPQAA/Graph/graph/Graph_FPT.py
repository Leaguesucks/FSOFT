import Prompt.Prompt as Prompt

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

from Retrieval.Storage import Storage, SearchResult

from LLM.Document.Summarizer import Candidate_Summary
from LLM.LLM import Model

from Tools.WebSearch import WebSearch

from Graph.state.State import FPTTaskState, ResultType, SearchType

class Graph_FPT:
    MAX_CHARS_PER_DOC = 5000

    def __init__(self, cb: Model=Model(), db: Storage=Storage(), 
                 ws: WebSearch=WebSearch()):
        self.cb = cb
        self.db = db
        self.ws = ws

        self.graph = StateGraph(FPTTaskState)

        self.graph.add_node("internal_search", self.search_internal_node)
        self.graph.add_node("web_search", self.search_web_node)
        self.graph.add_node("collect", self.collect_fpt_node)

        self.graph.add_edge(START, "internal_search")
        self.graph.add_edge(START, "web_search")

        self.graph.add_edge("internal_search", "collect")
        self.graph.add_edge("web_search", "collect")

        self.graph.add_edge("collect", END)

        self.graph = self.graph.compile()

    def emit_status(self, stage: str, message: str):
        """Display the current processing stage"""
        writer = get_stream_writer()

        writer({
            "type": "status",
            "stage": stage,
            "message": message
        })

    def build_context(self, results: list[SearchResult]) -> str:
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

                {root_payload["full_content"][:self.MAX_CHARS_PER_DOC]}
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
                if current_size + len(sub) > self.MAX_CHARS_PER_DOC:
                    break

                current_size += len(sub)
                source += sub
                sub_index += 1

            context_parts.append(source)

        return "\n".join(context_parts)

    def search_internal_node(self, state: FPTTaskState):
        task = state["task"]
        task_id = state["task_id"]

        self.emit_status(
            stage="internal_search",
            message=f"Searching FPT documents: {task}"
        )

        results = self.db.search(
            query=task,
            limit=5
        )

        if not results:
            context = ""
            sources = []
        else:
            context = self.build_context(results=results)

            sources = [
                result.payload.get("document_name", "")
                for result in results
            ]
            
        return {
            "search_results": {
                f"{task_id}: {SearchType.INTERNAL.value}":
                {
                    "content": context,
                    "sources": sources,
                }
            }
        }

    def search_web_node(self, state: FPTTaskState):
        task = state["task"]
        task_id = state["task_id"]

        self.emit_status(
            stage="web_search",
            message=f"Searching web: {task}"
        )

        context = self.ws.search(
            query=task,
            max_results=2,
            search_depth="advanced"
        )

        if len(context) > self.MAX_CHARS_PER_DOC:
            context = (
                context[:self.MAX_CHARS_PER_DOC] + 
                "\n[Webb search results truncated]"
            )

        return {
            "search_results": {
                f"{task_id}: {SearchType.WEB.value}":
                    {
                        "content": context,
                        "sources": []
                    }
            }
        }

    def collect_fpt_node(self, state: FPTTaskState):
        task = state["task"]
        task_id = state["task_id"]

        self.emit_status(
            stage="collecting_documents",
            message=f"Collecting doucuments: {task}"
        )

        search_results = state.get("search_results", {})

        internal = search_results.get(
            f"{task_id}: {SearchType.INTERNAL.value}",
            {}
        )

        web = search_results.get(
            f"{task_id}: {SearchType.WEB.value}",
            {}
        )

        internal_content = internal.get("content", "")
        web_content = web.get("content", "")

        prompt = f"""
            {Prompt.fpt_summarizer_prompt}

            USER'S QUERY:
            {task}

            INTERNAL CONTENT:
            {internal_content}

            EXTERNAL CONTENT:
            {web_content}

            CITE RULES:
            {Prompt.cite_rules}
        """

        content: Candidate_Summary = self.cb.summarizer.invoke(prompt)

        text = content.candidate
        text += f"INTERNAL:\n{content.internal_src}"

        if content.use_external:
            instruction = Prompt.fpt_internal_not_found_instruction
            text += f"EXTERNAL:\n{content.web_src}"
        else:
            instruction = Prompt.fpt_related_instruction

        task_result = {
            "task_id": task_id,
            "task": task,

            "result_type": ResultType.DOCUMENTS.value,

            "content": text,
            "error": "",

            "instruction": instruction
        }

        return {
            "task_results": [task_result]
        }