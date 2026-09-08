from tavily import TavilyClient
import os

class WebSearch:
    MAX_CHARS_PER_RESULT = 5000

    def __init__(self):
        self.client = TavilyClient(
            api_key=os.environ["TAVILY_API_KEY"]
        )

    def search(self, query: str, max_results: int=3, search_depth: str="basic") -> str:
        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=False,
            include_raw_content=True
        )

        results = response.get("results", [])

        if not results:
            return "No web search results were found."

        formatted = []
        for i, result in enumerate(results, start=1):
            content = result.get("content", "")

            if len(content) > self.MAX_CHARS_PER_RESULT:
                content = content[:self.MAX_CHARS_PER_RESULT] + "..."

            formatted.append(
                f"""
                    SOURCE {i}

                    TITLE:
                    {result.get("title", "")}

                    URL:
                    {result.get("url", "")}

                    CONTENT:
                    {content}
                """
            )

        return "\n\n".join(formatted)