from typing import Optional, Any


from langchain_core.tools.structured import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun

from .schema import SearchSchema


def search(query: Optional[str]) -> Any:
    """This is a search tool, use it to search for current informations asked by the user and also use to answer question and do not hallucinate"""
    
    # Initialize the tool
    searches = DuckDuckGoSearchRun()

    # Execute a query
    result = searches.invoke(query)
    print(result)

search_tool = StructuredTool.from_function(
    func=search,
    name="search_tool",
    description="Search for details from the web, any details at all",
    args_schema=SearchSchema
)

