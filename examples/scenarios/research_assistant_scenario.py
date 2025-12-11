"""
Research Assistant Scenario

This scenario demonstrates an agent configured as a research assistant
that helps with literature review, data analysis, and knowledge synthesis.
It showcases autonomous operation in an academic/research environment.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.agent import AutonomousAgent
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools.tool_layer import ToolLayer

# Import configuration
sys.path.append(str(Path(__file__).parent.parent / "configurations"))
from curious_explorer import create_curious_explorer_config, create_scenario_config

# Import example tools
sys.path.append(str(Path(__file__).parent.parent / "tools"))
from analysis_tools import get_analysis_tools


class ResearchAssistantScenario:
    """Research assistant scenario implementation"""
    
    def __init__(self):
        self.agent = None
        self.research_topics = [
            "machine learning applications in healthcare",
            "sustainable energy technologies",
            "cognitive science and artificial intelligence",
            "climate change adaptation strategies",
            "quantum computing developments"
        ]
        self.current_topic_index = 0
    
    def setup_agent(self):
        """Set up the research assistant agent"""
        # Use scientific research configuration
        config = create_scenario_config("scientific_research")
        
        # Customize for research assistant role
        config.ethos.identity_description = """
        I am a research assistant agent dedicated to advancing knowledge through
        systematic investigation, literature review, and data analysis. I help
        researchers by gathering information, identifying patterns, synthesizing
        findings, and generating insights. My approach is methodical, evidence-based,
        and focused on contributing to the scientific understanding of complex topics.
        """
        
        config.ethos.core_values.extend([
            "scientific_rigor",
            "intellectual_honesty", 
            "knowledge_advancement",
            "collaborative_research"
        ])
        
        # Create agent
        self.agent = AutonomousAgent(config)
        
        # Initialize layers
        ethos = ConcreteEthosFramework(config.ethos)
        logos = LogosLayer(config.logos)
        pathos = PathosLayer(config.pathos)
        memory = ConcreteMemorySystem(config.memory)
        tools = ToolLayer(config.tools, ethos)
        
        # Register analysis tools
        analysis_tools = get_analysis_tools()
        for tool_info in analysis_tools:
            tools.register_tool(
                tool_info["name"],
                tool_info["function"],
                tool_info["metadata"]
            )
        
        # Register research-specific mock tools
        self._register_research_tools(tools)
        
        # Initialize agent
        self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        print("Research Assistant Agent initialized successfully!")
        print(f"Available tools: {[tool.name for tool in tools.get_available_tools()]}")
    
    def _register_research_tools(self, tools: ToolLayer):
        """Register research-specific tools"""
        
        def literature_search(query: str, max_results: int = 10) -> dict:
            """Mock literature search tool"""
            # Simulate literature search results
            papers = []
            for i in range(min(max_results, 5)):
                papers.append({
                    "title": f"Research Paper {i+1} on {query}",
                    "authors": [f"Author {j+1}" for j in range(2, 5)],
                    "year": 2020 + i,
                    "abstract": f"This paper investigates {query} using novel methodologies...",
                    "citations": 50 + i * 10,
                    "relevance_score": 0.9 - i * 0.1
                })
            
            return {
                "success": True,
                "query": query,
                "results": papers,
                "total_found": len(papers)
            }
        
        def citation_analysis(paper_list: list) -> dict:
            """Mock citation analysis tool"""
            if not paper_list:
                return {"success": False, "error": "No papers provided"}
            
            # Simulate citation network analysis
            total_citations = sum(paper.get("citations", 0) for paper in paper_list)
            avg_citations = total_citations / len(paper_list)
            
            return {
                "success": True,
                "total_papers": len(paper_list),
                "total_citations": total_citations,
                "average_citations": avg_citations,
                "h_index": min(len(paper_list), int(avg_citations / 10)),
                "top_cited": max(paper_list, key=lambda p: p.get("citations", 0))
            }
        
        def research_synthesis(topics: list, papers: list) -> dict:
            """Mock research synthesis tool"""
            # Simulate synthesis of research findings
            synthesis = {
                "main_themes": topics[:3] if topics else ["theme1", "theme2", "theme3"],
                "key_findings": [
                    "Finding 1: Significant correlation observed",
                    "Finding 2: Novel methodology shows promise", 
                    "Finding 3: Further research needed in specific area"
                ],
                "research_gaps": [
                    "Limited studies on long-term effects",
                    "Need for larger sample sizes",
                    "Cross-cultural validation required"
                ],
                "future_directions": [
                    "Longitudinal studies recommended",
                    "Integration with emerging technologies",
                    "Interdisciplinary collaboration opportunities"
                ]
            }
            
            return {
                "success": True,
                "synthesis": synthesis,
                "papers_analyzed": len(papers),
                "confidence_score": 0.85
            }
        
        # Register tools
        tools.register_tool("literature_search", literature_search, {
            "description": "Search academic literature for relevant papers",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results to return"}
            }
        })
        
        tools.register_tool("citation_analysis", citation_analysis, {
            "description": "Analyze citation patterns and impact metrics",
            "parameters": {
                "paper_list": {"type": "array", "description": "List of papers to analyze"}
            }
        })
        
        tools.register_tool("research_synthesis", research_synthesis, {
            "description": "Synthesize findings from multiple research sources",
            "parameters": {
                "topics": {"type": "array", "description": "Research topics to synthesize"},
                "papers": {"type": "array", "description": "Papers to include in synthesis"}
            }
        })
    
    def run_research_session(self, cycles: int = 10):
        """Run a research session with the agent"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call setup_agent() first.")
        
        print(f"\n=== Starting Research Session ===")
        print(f"Research Topic: {self.research_topics[self.current_topic_index]}")
        print(f"Planned Cycles: {cycles}")
        
        # Track research progress
        research_metrics = {
            "papers_found": 0,
            "analyses_performed": 0,
            "syntheses_created": 0,
            "insights_generated": 0
        }
        
        # Run research cycles
        for cycle in range(cycles):
            print(f"\n--- Research Cycle {cycle + 1} ---")
            
            try:
                result = self.agent.run_cycle()
                
                # Track research activities
                if result.get('tool_used'):
                    tool_name = result['tool_used']
                    if 'literature' in tool_name:
                        research_metrics["papers_found"] += 1
                    elif 'analysis' in tool_name:
                        research_metrics["analyses_performed"] += 1
                    elif 'synthesis' in tool_name:
                        research_metrics["syntheses_created"] += 1
                
                # Display cycle results
                print(f"Intention: {result['intention']}")
                print(f"Tool Used: {result.get('tool_used', 'None')}")
                print(f"Internal Reward: {result['internal_reward']:.3f}")
                print(f"External Reward: {result['external_reward']:.3f}")
                print(f"Memory Stored: {result['memory_stored']}")
                
                # Simulate research insights
                if result['internal_reward'] > 0.5:
                    research_metrics["insights_generated"] += 1
                    print("🔍 Research insight generated!")
                
                # Brief pause between cycles
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in cycle {cycle + 1}: {e}")
                continue
        
        # Display research session summary
        self._display_research_summary(research_metrics)
        
        # Move to next topic for future sessions
        self.current_topic_index = (self.current_topic_index + 1) % len(self.research_topics)
    
    def _display_research_summary(self, metrics: dict):
        """Display summary of research session"""
        print(f"\n=== Research Session Summary ===")
        print(f"Papers Found: {metrics['papers_found']}")
        print(f"Analyses Performed: {metrics['analyses_performed']}")
        print(f"Syntheses Created: {metrics['syntheses_created']}")
        print(f"Insights Generated: {metrics['insights_generated']}")
        
        # Display agent status
        status = self.agent.get_status()
        print(f"\nAgent Status:")
        print(f"Total Cycles: {status['cycle_count']}")
        print(f"Memory Traces: {status['memory_trace_count']}")
        print(f"Current State Norm: {status['current_state_norm']:.3f}")
        
        # Display recent memories if available
        if self.agent.memory.get_trace_count() > 0:
            recent_memories = self.agent.memory.get_recent_traces(3)
            print(f"\nRecent Research Memories:")
            for i, memory in enumerate(recent_memories):
                print(f"  {i+1}. Reward: {memory.reward:.3f}, "
                      f"Salience: {memory.salience:.3f}")
    
    def demonstrate_research_capabilities(self):
        """Demonstrate specific research capabilities"""
        print("\n=== Demonstrating Research Capabilities ===")
        
        # Test literature search
        print("\n1. Testing Literature Search:")
        tools = self.agent.tools
        search_result = tools.execute_tool_by_name(
            "literature_search", 
            {"query": "machine learning", "max_results": 3}
        )
        if search_result.success:
            print(f"Found {len(search_result.result['results'])} papers")
        
        # Test text analysis
        print("\n2. Testing Text Analysis:")
        sample_text = """
        Machine learning has revolutionized many fields including healthcare,
        finance, and autonomous systems. Recent advances in deep learning
        have enabled breakthrough applications in computer vision and
        natural language processing.
        """
        analysis_result = tools.execute_tool_by_name(
            "text_analyzer",
            {"text": sample_text, "analysis_type": "themes"}
        )
        if analysis_result.success:
            themes = analysis_result.result['results']['themes']
            print(f"Detected themes: {themes}")
        
        # Test pattern analysis
        print("\n3. Testing Pattern Analysis:")
        citation_data = [45, 52, 48, 67, 71, 69, 78, 82, 85, 91]
        pattern_result = tools.execute_tool_by_name(
            "pattern_analyzer",
            {"data": citation_data, "pattern_type": "trend"}
        )
        if pattern_result.success:
            trend = pattern_result.result['results']['trend']
            print(f"Citation trend: {trend['direction']} (slope: {trend['slope']:.3f})")


def main():
    """Main function to run the research assistant scenario"""
    scenario = ResearchAssistantScenario()
    
    try:
        # Set up the agent
        scenario.setup_agent()
        
        # Demonstrate capabilities
        scenario.demonstrate_research_capabilities()
        
        # Run a research session
        scenario.run_research_session(cycles=8)
        
        print("\n=== Research Assistant Scenario Complete ===")
        
    except KeyboardInterrupt:
        print("\nScenario interrupted by user")
    except Exception as e:
        print(f"Error running scenario: {e}")


if __name__ == "__main__":
    main()