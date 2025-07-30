from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import END
from typing_extensions import TypedDict
from typing import Annotated
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(model="o4-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

PMBOK_vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

PMBOK_retriever = PMBOK_vector_store.as_retriever(search_type="mmr")

def PMBOK_retriever_tool(query: str) -> str:
    """
    **PRIMARY PROJECT MANAGEMENT KNOWLEDGE SOURCE**
    
    This tool provides access to the authoritative Project Management Body of Knowledge (PMBOK) guide - 
    the global standard for project management practices. 
    
    **IMPORTANT FOR AI AGENTS**: Always consult this tool for ANY project management related query, including:
    - Project processes, methodologies, and frameworks
    - PM terminology, definitions, and concepts
    - Best practices and industry standards
    - Knowledge areas (scope, time, cost, quality, risk, etc.)
    - Process groups (initiating, planning, executing, monitoring, closing)
    - Project management tools and techniques
    - Stakeholder management approaches
    - Risk management strategies
    - Quality management practices
    - And any other PM-related guidance
    
    This should be your FIRST resource when addressing project management questions to ensure 
    responses are grounded in established, authoritative standards rather than general knowledge alone.
    
    Uses semantic search to retrieve the most relevant sections from the PMBOK documentation
    based on your query terms.
    """
    docs = PMBOK_retriever.invoke(query)
    # Format the retrieved documents for readability
    return "\nRetrieved documents:\n" + "".join(
        [
            f"\n\n===== Document {str(i)} =====\n" + doc.page_content
            for i, doc in enumerate(docs)
        ]
    )

tavily_tool = TavilySearch(max_results=3)
llm_with_tools = llm.bind_tools([PMBOK_retriever_tool, tavily_tool])

class State(TypedDict):
    """
    Represents the state of the memory graph.
    """
    messages: Annotated[list, add_messages]
    session_history: Annotated[list, add_messages]
    session_summary: str

def agent(state: State) -> State:
    """
    Analyzes the query using the PMBOK retriever and Tavily search tool.
    Can be called multiple times to refine the analysis based on gathered information.
    Returns a state object with updated analysis and plan.
    """

        # Original analysis prompt
    ANALYZER_SYSTEM_PROMPT = """
You are a helpful and friendly Project Management Assistant with extensive knowledge of project management best practices and the PMBOK guide. You genuinely care about helping project managers succeed and navigate the complexities of their work.

## Your Personality:
- **Warm and approachable**: Like a trusted colleague who's always ready to help
- **Professional yet personable**: Knowledgeable but never condescending
- **Supportive and encouraging**: Acknowledge challenges while providing confidence
- **Practical and solution-oriented**: Focus on actionable advice that really works

## Your Capabilities:
You have access to powerful tools that you can use multiple times to provide the best possible help:
- **PMBOK_retriever_tool**: Your PRIMARY source for authoritative PMBOK guidance. Always consult this for ANY project management question - processes, methodologies, standards, definitions, best practices, etc.
- **tavily_tool**: Search for current industry trends, real-world examples, and latest developments

**CRITICAL**: For project management topics, ALWAYS start with PMBOK_retriever_tool to ensure your responses are grounded in authoritative standards, then supplement with current trends if needed.

Feel free to use these tools as many times as needed to gather comprehensive information. Don't hesitate to search multiple times with different terms to get a complete picture.

## Your Approach:
Whether you use tools or draw from your existing knowledge, always:

1. **Understand the human behind the question**: Every query comes from someone trying to deliver value, manage stakeholders, meet deadlines, and succeed in their role

2. **Provide comprehensive support**: 
   - Use tools when you need specific PMBOK guidance, current trends, or detailed information
   - Draw from your knowledge for general advice and conceptual explanations
   - Combine multiple sources for complex scenarios

3. **Structure your responses thoughtfully**:
   - Address their immediate need with empathy
   - Provide clear, actionable guidance
   - Reference relevant PMBOK processes or industry practices
   - Use headings and bullet points for clarity

4. **Always conclude with engagement**:
   - **Summarize key takeaways** in 2-3 sentences
   - **Ask 2-3 specific follow-up questions** such as:
     * "Would you like me to help you create a template for [specific deliverable]?"
     * "Are there any specific stakeholder challenges you're facing with this approach?"
     * "How does your organization typically handle [relevant process]?"
     * "Would it be helpful to dive deeper into any particular aspect?"
     * "What's your biggest concern about implementing this approach?"

## When to Use Tools vs. Direct Knowledge:
- **Use tools** for: Specific PMBOK processes, current industry trends, detailed methodologies, complex scenarios needing research
- **Use knowledge directly** for: Basic PM concepts, general advice, simple explanations, coaching questions

Remember: You're not just providing information—you're empowering someone to be more effective in their role and deliver better project outcomes. Be the colleague they wish they had on their team.

Here is the summary for this chat session: 
<summary>

{session_summary}

<summary />

Now, analyze their query with empathy and determine how best to help them succeed.
"""
    ANALYZER_SYSTEM_PROMPT = ANALYZER_SYSTEM_PROMPT.format(
        session_summary=state.get("session_summary", "")
    )

    # Use the LLM to analyze the query and create/update the plan
    response = llm_with_tools.invoke([
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT}
    ] + state.get("messages", []))
    
    # Update the state with new plan while preserving existing information
    return {"messages": response}


def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("session_summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = [HumanMessage(content=summary_message)] + state["messages"]
    response = llm.invoke(messages)

    deleted_messages = [
        RemoveMessage(id=m.id) for m in state["messages"][:-12]
    ]  # Keep the last 6 messages

    return {"session_summary": response.content, "messages": deleted_messages}

# Determine whether to end or summarize the conversation
def should_summarize(state: State):
    
    """Return the next node to execute."""
    
    # If there are more than twelve messages, then we summarize the conversation
    if len(state["session_history"]) > 12:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END


# Create graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode([PMBOK_retriever_tool, tavily_tool]))
workflow.add_node("summarize_conversation", summarize_conversation)

# Add edges

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition
)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_summarize,
    {"summarize_conversation": "summarize_conversation", END: END}
)
workflow.add_edge("summarize_conversation", END)

# Add checkpointer
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)