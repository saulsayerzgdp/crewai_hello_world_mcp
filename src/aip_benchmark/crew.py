from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import MCPServerAdapter  # Import MCPServerAdapter for arXiv tool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Connect to MCP SSE server for arXiv tools
server_params = {
    "url": "https://mcp.obrol.id/f/sse",
    "transport": "sse"
}

# Global adapter and tools
mcp_server_adapter = MCPServerAdapter(server_params)
mcp_tools = None

@CrewBase
class AipBenchmark():
    """AipBenchmark crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def arxiv_research_agent(self) -> Agent:
        """
        Agent for searching arXiv papers using tools from MCP SSE server.
        """
        print("AVAILABLE TOOLS:", mcp_tools)
        return Agent(
            config=self.agents_config['arxiv_research_agent'], # type: ignore[index]
            tools=mcp_tools,
            verbose=True,
            reasoning=True
        )

    @task
    def arxiv_research_task(self) -> Task:
        """
        Task for searching arXiv papers by topic and number of items using the arxiv_research_agent.
        """
        # The agent will have the MCP tools attached automatically
        return Task(
            config=self.tasks_config['arxiv_research_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HelloWorld crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )

def start_mcp_adapter():
    global mcp_tools
    try:
        mcp_server_adapter.start()
        mcp_tools = mcp_server_adapter.tools
        print(f"Available tools: {[tool.name for tool in mcp_tools]}")
    except Exception as e:
        print(f"Error starting MCP adapter: {e}")
        mcp_tools = mcp_server_adapter.tools

def stop_mcp_adapter():
    if mcp_server_adapter and getattr(mcp_server_adapter, 'is_connected', False):
        print("Stopping MCP server connection…")
        mcp_server_adapter.stop()
    elif mcp_server_adapter:
        print("MCP server adapter was not connected. No stop needed or start failed.")