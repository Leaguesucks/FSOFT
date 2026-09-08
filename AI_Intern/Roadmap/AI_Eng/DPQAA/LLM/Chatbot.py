import httpx

from langchain_openai import ChatOpenAI as ChatModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_client.models import ScoredPoint

from Retrieval.Storage import Storage
from LLM.RAG import RAG
from LLM.QueryRoute import QueryRoute, QueryType
from LLM.QueryResolver import ResolvedQuery
from Tools.CodeExecutor import Language
from Tools.WebSearch import WebSearch

class Chatbot:
    '''Responsible for genereating a RAG pipeline'''
    MAX_WEB_CONTEXT_CHARS = 15000

    def __init__(self, db: Storage, llm: ChatModel):
        self.db = db
        self.llm = llm
        self.web_search = WebSearch()
        self.router = llm.with_structured_output(QueryRoute)
        self.query_resolver = llm.with_structured_output(ResolvedQuery)

        self.store = {}

        self.prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
            You are an internal FPT Software policy assistant.

            You must follow these rules:

            {RAG.rules}

            Stylish your answers using HTML follwing these rules:
            {RAG.markdown_rules}
            """
        ),

        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            """
            Retrieved document context. Ignore this if the user query is casual (e.g., not a query that need searching):

            {context}

            User question:

            {query}

            Again, no reference if the user question is CASUAL.

            You may look at previous questions and answers for broader context.
            """
        )])

        self.chain = self.prompt | self.llm

        self.conversation = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self.get_session_history,
            input_messages_key="query",
            history_messages_key="history"
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = (InMemoryChatMessageHistory())

        return self.store[session_id]

    def get_history_text(self, session_id: str="user_123") -> str:
        history = self.get_session_history(session_id=session_id)
        parts = []

        for message in history.messages:
            parts.append(f"{message.type}: {message.content}")

        return "\n".join(parts)

    def resolve_query(self, query: str, session_id: str="user_123") -> ResolvedQuery:
        history = self.get_history_text(session_id=session_id)
        prompt = f"""
            {RAG.history_resolver_prompt}

            CONVERSATION:
            {history}

            USER's LATEST MESSAGE:
            {query}
        """

        return self.query_resolver.invoke(prompt)

    def retrieve(self, query: str, limit: int=5) -> list[ScoredPoint]:
        '''Search the vector database for the query'''
        return self.db.search(
            query=query,
            limit=limit
        )

    def prepare_query(self, query: str, limit: int, min_score: float) -> str:
        results = self.retrieve(query=query, limit=limit)
        relevant_results = [result for result in results if result.score >= min_score]
        return self.build_context(results=relevant_results)

    def classify_query(self, query: str) -> QueryRoute:
        prompt = f"""
        {RAG.router_prompt}

        USER QUERY:
        {query}
        """
        return self.router.invoke(prompt)

    def build_context(self, results: list[ScoredPoint], max_chars: int=20000) -> str:
        '''Build full context based on the retrieved answer'''

        # if not results:
        #     return RAG.non_fpt_prompt

        context_parts = []
        current_size = 0

        for i, result in enumerate(results, start=1):
            payload = result.payload
            source = f"""
                --- SOURCE {i} ---
                Document: {payload["document_name"]}
                Section: {payload["title"]}
                Page(s): {payload["pages"]}

                {payload["content"]}
            """

            if current_size + len(source) > max_chars:
                break

            context_parts.append(source)
            current_size += len(source)

        return "\n".join(context_parts)

    def execute_python(self, code: str):
        response = httpx.post(
            "http://127.0.0.1:8000/execute",
            json={
                "language": Language.PYTHON.value,
                "code": code
            },
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    def write_code_and_compute(self, query: str) -> str:
        prompt = f" {RAG.code_solver_instruction}\n User Query:\n {query}"
        code = self.llm.invoke(prompt).content

        execution = self.execute_python(code)

        if execution["exit_code"] != 0:
            print(execution["stderr"])
            return f"Say {RAG.rejection}"

        return f"Say here is the answer to the query: {execution["stdout"]}\n User Query:\n {query}"

    # def answer(self, query: str, limit: int=5, min_score=0.50, session_id="user_123") -> str:
    #     '''Answer to the query using RAG pipeline'''
    #     context = self.prepare_query(query=query, limit=limit, min_score=min_score)

    #     response = self.conversation.invoke(
    #         {
    #             "query": query,
    #             "context": context,
    #         },
    #         config={
    #             "configurable": {
    #                 "session_id": session_id
    #             }
    #         }
    #     )
    #     return response.content

    def answer_stream(self, query: str, limit: int=5, min_score: float=0.50, session_id:str="user_123"):
        '''Stream the answer token-by-token'''
        resolved = self.resolve_query(query=query, session_id=session_id)
        resolved_query = resolved.query
        previous_context = resolved.context

        route = self.classify_query(query=resolved_query)
        route_type = route.query_type

        if route_type == QueryType.FPT:
            route_instruction = RAG.fpt_related_instruction
            retrieved_context = self.prepare_query(query=resolved_query, limit=limit, min_score=min_score)
        elif route_type == QueryType.DOCUMENT:
            route_instruction = RAG.non_fpt_search_instruction
            retrieved_context = self.web_search.search(
                query=resolved_query,
                max_results=5,
                search_depth="basic"
            )

            if len(retrieved_context) > self.MAX_WEB_CONTEXT_CHARS: # Hard-code chunking, improve later
                retrieved_context = (retrieved_context[:self.MAX_WEB_CONTEXT_CHARS]
                                     + "\n[Web search results truncated]")
        elif route_type == QueryType.COMPUTE:
            route_instruction = "Answer the user's computational question using the provided computed result"
            retrieved_context = self.write_code_and_compute(query=resolved_query)
        elif route_type == QueryType.CODE:
            route_instruction = RAG.code_instruction
            retrieved_context = ""
        elif route_type == QueryType.RUBBISH:
            route_instruction = RAG.rubbish_instruction
            retrieved_context = ""
        elif route_type == QueryType.LACK_CONTEXT:
            route_instruction = RAG.lack_context_instruction
            retrieved_context = ""
        elif route_type == QueryType.GENERAL:
            route_instruction = RAG.casual_instruction
            retrieved_context = ""
        else:
            return "I do not understand this query"

        context = f"""
            ROUTING INSTRUCTION:
            {route_instruction}

            RETRIEVED DATA:
            {retrieved_context}

            CURRENT QUERY:
            {resolved_query}

            PREVIOUS CONVERSATION CONTEXT:
            {previous_context}
        """

        # print(context, route_type) # Debug
        # print(retrieved_context)

        for chunk in self.conversation.stream(
            {
                "query": query,
                "context": context
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        ):
            if chunk.content:
                yield chunk.content

