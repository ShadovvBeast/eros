"""
Creative Writing Scenario

This scenario demonstrates an agent configured for creative writing tasks,
including story generation, character development, and narrative analysis.
It showcases autonomous creative processes and collaborative writing.
"""

import sys
import os
import time
import random
from pathlib import Path
from typing import List, Dict, Any

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
from creative_collaborator import create_creative_collaborator_config, create_creative_scenario_config


class CreativeWritingScenario:
    """Creative writing scenario implementation"""
    
    def __init__(self):
        self.agent = None
        self.story_elements = {
            "genres": ["science fiction", "fantasy", "mystery", "romance", "thriller", "historical fiction"],
            "settings": ["distant planet", "medieval castle", "modern city", "space station", "enchanted forest", "underwater city"],
            "character_types": ["reluctant hero", "wise mentor", "mysterious stranger", "loyal companion", "cunning antagonist"],
            "themes": ["redemption", "discovery", "love conquers all", "power corrupts", "coming of age", "sacrifice"]
        }
        self.current_story = {
            "title": "",
            "genre": "",
            "characters": [],
            "plot_points": [],
            "themes": []
        }
    
    def setup_agent(self):
        """Set up the creative writing agent"""
        # Use storytelling configuration
        config = create_creative_scenario_config("storytelling")
        
        # Customize for creative writing
        config.ethos.identity_description = """
        I am a creative writing agent passionate about storytelling, character development,
        and narrative craft. I help writers explore ideas, develop compelling characters,
        and create engaging stories across various genres. My approach combines creativity
        with structural understanding of narrative techniques, always seeking to create
        emotionally resonant and meaningful stories.
        """
        
        config.ethos.core_values.extend([
            "storytelling_excellence",
            "character_authenticity",
            "emotional_resonance",
            "creative_exploration"
        ])
        
        # Create agent
        self.agent = AutonomousAgent(config)
        
        # Initialize layers
        ethos = ConcreteEthosFramework(config.ethos)
        logos = LogosLayer(config.logos)
        pathos = PathosLayer(config.pathos)
        memory = ConcreteMemorySystem(config.memory)
        tools = ToolLayer(config.tools, ethos)
        
        # Register creative writing tools
        self._register_writing_tools(tools)
        
        # Initialize agent
        self.agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        print("Creative Writing Agent initialized successfully!")
        print(f"Available tools: {[tool.name for tool in tools.get_available_tools()]}")
    
    def _register_writing_tools(self, tools: ToolLayer):
        """Register creative writing specific tools"""
        
        def character_generator(character_type: str = "random", genre: str = "general") -> dict:
            """Generate character profiles"""
            names = ["Alex", "Morgan", "Casey", "Jordan", "Riley", "Sage", "Phoenix", "River"]
            traits = ["brave", "curious", "cautious", "ambitious", "loyal", "mysterious", "witty", "compassionate"]
            backgrounds = ["noble", "commoner", "scholar", "warrior", "merchant", "artist", "outcast", "leader"]
            
            character = {
                "name": random.choice(names),
                "type": character_type if character_type != "random" else random.choice(self.story_elements["character_types"]),
                "traits": random.sample(traits, 3),
                "background": random.choice(backgrounds),
                "motivation": f"Seeks to {random.choice(['find truth', 'protect others', 'gain power', 'find love', 'discover identity'])}",
                "conflict": f"Struggles with {random.choice(['past mistakes', 'divided loyalty', 'hidden identity', 'moral dilemma', 'fear of failure'])}",
                "genre_fit": genre
            }
            
            return {
                "success": True,
                "character": character,
                "description": f"Generated {character['type']} character: {character['name']}"
            }
        
        def plot_generator(genre: str = "general", complexity: str = "medium") -> dict:
            """Generate plot structures and story beats"""
            
            # Basic three-act structure
            act1_beats = [
                "Introduce protagonist in ordinary world",
                "Present inciting incident",
                "Establish stakes and goals",
                "Cross threshold into adventure"
            ]
            
            act2_beats = [
                "Face initial challenges",
                "Meet allies and enemies", 
                "Encounter major obstacle",
                "Experience crisis of faith",
                "Discover hidden truth",
                "Prepare for final confrontation"
            ]
            
            act3_beats = [
                "Face ultimate challenge",
                "Make crucial sacrifice or choice",
                "Achieve resolution",
                "Return transformed"
            ]
            
            plot_structure = {
                "act_1": act1_beats,
                "act_2": act2_beats[:4] if complexity == "simple" else act2_beats,
                "act_3": act3_beats,
                "genre": genre,
                "complexity": complexity
            }
            
            return {
                "success": True,
                "plot_structure": plot_structure,
                "total_beats": len(act1_beats) + len(plot_structure["act_2"]) + len(act3_beats)
            }
        
        def dialogue_enhancer(dialogue: str, character_voice: str = "neutral") -> dict:
            """Enhance dialogue with character voice and style"""
            
            # Simple dialogue enhancement (mock implementation)
            voice_modifiers = {
                "formal": lambda text: text.replace("don't", "do not").replace("can't", "cannot"),
                "casual": lambda text: text.replace("do not", "don't").replace("cannot", "can't"),
                "archaic": lambda text: f"Verily, {text}",
                "technical": lambda text: f"{text} (speaking precisely)",
                "emotional": lambda text: f"{text}!"
            }
            
            enhanced = dialogue
            if character_voice in voice_modifiers:
                enhanced = voice_modifiers[character_voice](dialogue)
            
            return {
                "success": True,
                "original": dialogue,
                "enhanced": enhanced,
                "voice": character_voice,
                "improvements": ["Applied character voice", "Enhanced authenticity"]
            }
        
        def story_analyzer(text: str, analysis_type: str = "structure") -> dict:
            """Analyze story elements and structure"""
            
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            analysis = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": word_count / max(sentence_count, 1),
                "estimated_reading_time": word_count / 200  # words per minute
            }
            
            if analysis_type == "structure":
                # Simple structure analysis
                analysis["structure"] = {
                    "has_dialogue": '"' in text or "'" in text,
                    "has_action": any(word in text.lower() for word in ["ran", "jumped", "fought", "moved"]),
                    "has_description": any(word in text.lower() for word in ["beautiful", "dark", "bright", "cold"]),
                    "narrative_style": "first_person" if " I " in text else "third_person"
                }
            
            elif analysis_type == "themes":
                # Simple theme detection
                theme_keywords = {
                    "love": ["love", "heart", "romance", "affection"],
                    "conflict": ["fight", "battle", "struggle", "conflict"],
                    "mystery": ["secret", "hidden", "mystery", "unknown"],
                    "growth": ["learn", "grow", "change", "transform"]
                }
                
                detected_themes = []
                for theme, keywords in theme_keywords.items():
                    if any(keyword in text.lower() for keyword in keywords):
                        detected_themes.append(theme)
                
                analysis["themes"] = detected_themes
            
            return {
                "success": True,
                "analysis": analysis,
                "analysis_type": analysis_type
            }
        
        def world_builder(setting_type: str = "fantasy", detail_level: str = "medium") -> dict:
            """Generate world-building elements"""
            
            world_elements = {
                "fantasy": {
                    "locations": ["Enchanted Forest", "Crystal Caves", "Floating Islands", "Ancient Ruins"],
                    "cultures": ["Elven Kingdoms", "Dwarven Clans", "Human Empires", "Nomadic Tribes"],
                    "magic_system": "Elemental magic based on natural forces",
                    "conflicts": ["Ancient evil awakening", "War between kingdoms", "Magic vs technology"]
                },
                "sci_fi": {
                    "locations": ["Space Stations", "Alien Worlds", "Generation Ships", "Terraformed Planets"],
                    "cultures": ["Galactic Federation", "AI Collectives", "Nomadic Fleets", "Corporate States"],
                    "technology": "Faster-than-light travel and quantum computing",
                    "conflicts": ["First contact scenarios", "AI uprising", "Resource wars"]
                },
                "modern": {
                    "locations": ["Metropolitan Cities", "Small Towns", "Corporate Offices", "Underground Networks"],
                    "cultures": ["Tech Entrepreneurs", "Academic Communities", "Artist Collectives", "Political Movements"],
                    "technology": "Advanced internet and biotechnology",
                    "conflicts": ["Social inequality", "Environmental crisis", "Political corruption"]
                }
            }
            
            world_type = setting_type if setting_type in world_elements else "fantasy"
            world_data = world_elements[world_type]
            
            # Select elements based on detail level
            num_elements = {"simple": 2, "medium": 3, "complex": 4}[detail_level]
            
            selected_world = {
                "type": world_type,
                "locations": world_data["locations"][:num_elements],
                "cultures": world_data["cultures"][:num_elements],
                "key_feature": list(world_data.keys())[2],  # magic_system, technology, etc.
                "main_conflict": random.choice(world_data["conflicts"])
            }
            
            return {
                "success": True,
                "world": selected_world,
                "detail_level": detail_level
            }
        
        # Register all tools
        tools.register_tool("character_generator", character_generator, {
            "description": "Generate detailed character profiles for stories",
            "parameters": {
                "character_type": {"type": "string", "description": "Type of character to generate"},
                "genre": {"type": "string", "description": "Genre context for character"}
            }
        })
        
        tools.register_tool("plot_generator", plot_generator, {
            "description": "Generate plot structures and story beats",
            "parameters": {
                "genre": {"type": "string", "description": "Story genre"},
                "complexity": {"type": "string", "description": "Plot complexity level"}
            }
        })
        
        tools.register_tool("dialogue_enhancer", dialogue_enhancer, {
            "description": "Enhance dialogue with character voice and style",
            "parameters": {
                "dialogue": {"type": "string", "description": "Dialogue to enhance"},
                "character_voice": {"type": "string", "description": "Character voice style"}
            }
        })
        
        tools.register_tool("story_analyzer", story_analyzer, {
            "description": "Analyze story elements and structure",
            "parameters": {
                "text": {"type": "string", "description": "Story text to analyze"},
                "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
            }
        })
        
        tools.register_tool("world_builder", world_builder, {
            "description": "Generate world-building elements and settings",
            "parameters": {
                "setting_type": {"type": "string", "description": "Type of world setting"},
                "detail_level": {"type": "string", "description": "Level of detail for world"}
            }
        })
    
    def run_creative_session(self, cycles: int = 12):
        """Run a creative writing session with the agent"""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call setup_agent() first.")
        
        print(f"\n=== Starting Creative Writing Session ===")
        print(f"Planned Cycles: {cycles}")
        
        # Initialize story project
        self._initialize_story_project()
        
        # Track creative metrics
        creative_metrics = {
            "characters_created": 0,
            "plot_elements_generated": 0,
            "dialogue_enhanced": 0,
            "world_elements_built": 0,
            "creative_insights": 0
        }
        
        # Run creative cycles
        for cycle in range(cycles):
            print(f"\n--- Creative Cycle {cycle + 1} ---")
            
            try:
                result = self.agent.run_cycle()
                
                # Track creative activities
                if result.get('tool_used'):
                    tool_name = result['tool_used']
                    if 'character' in tool_name:
                        creative_metrics["characters_created"] += 1
                    elif 'plot' in tool_name:
                        creative_metrics["plot_elements_generated"] += 1
                    elif 'dialogue' in tool_name:
                        creative_metrics["dialogue_enhanced"] += 1
                    elif 'world' in tool_name:
                        creative_metrics["world_elements_built"] += 1
                
                # Display cycle results
                print(f"Creative Intention: {result['intention']}")
                print(f"Tool Used: {result.get('tool_used', 'None')}")
                print(f"Internal Reward: {result['internal_reward']:.3f}")
                print(f"External Reward: {result['external_reward']:.3f}")
                print(f"Memory Stored: {result['memory_stored']}")
                
                # Simulate creative insights
                if result['internal_reward'] > 0.6:
                    creative_metrics["creative_insights"] += 1
                    print("✨ Creative insight generated!")
                
                # Brief pause between cycles
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in cycle {cycle + 1}: {e}")
                continue
        
        # Display creative session summary
        self._display_creative_summary(creative_metrics)
    
    def _initialize_story_project(self):
        """Initialize a new story project"""
        self.current_story = {
            "title": "The Autonomous Tale",
            "genre": random.choice(self.story_elements["genres"]),
            "setting": random.choice(self.story_elements["settings"]),
            "theme": random.choice(self.story_elements["themes"]),
            "characters": [],
            "plot_points": []
        }
        
        print(f"Story Project Initialized:")
        print(f"  Title: {self.current_story['title']}")
        print(f"  Genre: {self.current_story['genre']}")
        print(f"  Setting: {self.current_story['setting']}")
        print(f"  Theme: {self.current_story['theme']}")
    
    def _display_creative_summary(self, metrics: dict):
        """Display summary of creative session"""
        print(f"\n=== Creative Writing Session Summary ===")
        print(f"Characters Created: {metrics['characters_created']}")
        print(f"Plot Elements Generated: {metrics['plot_elements_generated']}")
        print(f"Dialogue Enhanced: {metrics['dialogue_enhanced']}")
        print(f"World Elements Built: {metrics['world_elements_built']}")
        print(f"Creative Insights: {metrics['creative_insights']}")
        
        # Display agent status
        status = self.agent.get_status()
        print(f"\nAgent Status:")
        print(f"Total Cycles: {status['cycle_count']}")
        print(f"Memory Traces: {status['memory_trace_count']}")
        print(f"Current State Norm: {status['current_state_norm']:.3f}")
        
        # Display story project status
        print(f"\nStory Project: {self.current_story['title']}")
        print(f"Genre: {self.current_story['genre']}")
        print(f"Characters: {len(self.current_story['characters'])}")
        print(f"Plot Points: {len(self.current_story['plot_points'])}")
    
    def demonstrate_writing_capabilities(self):
        """Demonstrate specific creative writing capabilities"""
        print("\n=== Demonstrating Creative Writing Capabilities ===")
        
        tools = self.agent.tools
        
        # Test character generation
        print("\n1. Testing Character Generation:")
        char_result = tools.execute_tool_by_name(
            "character_generator",
            {"character_type": "reluctant hero", "genre": "fantasy"}
        )
        if char_result.success:
            character = char_result.result['character']
            print(f"Generated character: {character['name']} ({character['type']})")
            print(f"Traits: {', '.join(character['traits'])}")
        
        # Test plot generation
        print("\n2. Testing Plot Generation:")
        plot_result = tools.execute_tool_by_name(
            "plot_generator",
            {"genre": "science fiction", "complexity": "medium"}
        )
        if plot_result.success:
            plot = plot_result.result['plot_structure']
            print(f"Generated {plot['complexity']} {plot['genre']} plot")
            print(f"Total story beats: {plot_result.result['total_beats']}")
        
        # Test dialogue enhancement
        print("\n3. Testing Dialogue Enhancement:")
        dialogue_result = tools.execute_tool_by_name(
            "dialogue_enhancer",
            {"dialogue": "I don't think we can do this.", "character_voice": "formal"}
        )
        if dialogue_result.success:
            print(f"Original: {dialogue_result.result['original']}")
            print(f"Enhanced: {dialogue_result.result['enhanced']}")
        
        # Test world building
        print("\n4. Testing World Building:")
        world_result = tools.execute_tool_by_name(
            "world_builder",
            {"setting_type": "fantasy", "detail_level": "medium"}
        )
        if world_result.success:
            world = world_result.result['world']
            print(f"Generated {world['type']} world")
            print(f"Locations: {', '.join(world['locations'])}")
            print(f"Main conflict: {world['main_conflict']}")
    
    def collaborative_writing_demo(self):
        """Demonstrate collaborative writing features"""
        print("\n=== Collaborative Writing Demo ===")
        
        # Simulate collaborative story development
        story_prompt = "In a world where memories can be traded like currency..."
        
        print(f"Story Prompt: {story_prompt}")
        
        tools = self.agent.tools
        
        # Generate world for the prompt
        world_result = tools.execute_tool_by_name(
            "world_builder",
            {"setting_type": "sci_fi", "detail_level": "complex"}
        )
        
        if world_result.success:
            world = world_result.result['world']
            print(f"\nGenerated World Context:")
            print(f"Setting: {world['type']}")
            print(f"Key Feature: {world.get('key_feature', 'Advanced technology')}")
            print(f"Main Conflict: {world['main_conflict']}")
        
        # Generate protagonist
        char_result = tools.execute_tool_by_name(
            "character_generator",
            {"character_type": "reluctant hero", "genre": "sci_fi"}
        )
        
        if char_result.success:
            character = char_result.result['character']
            print(f"\nProtagonist: {character['name']}")
            print(f"Background: {character['background']}")
            print(f"Motivation: {character['motivation']}")
            print(f"Conflict: {character['conflict']}")
        
        # Generate plot structure
        plot_result = tools.execute_tool_by_name(
            "plot_generator",
            {"genre": "sci_fi", "complexity": "complex"}
        )
        
        if plot_result.success:
            print(f"\nStory Structure Generated:")
            print(f"Total beats: {plot_result.result['total_beats']}")
            print("Ready for collaborative development!")


def main():
    """Main function to run the creative writing scenario"""
    scenario = CreativeWritingScenario()
    
    try:
        # Set up the agent
        scenario.setup_agent()
        
        # Demonstrate capabilities
        scenario.demonstrate_writing_capabilities()
        
        # Show collaborative features
        scenario.collaborative_writing_demo()
        
        # Run a creative session
        scenario.run_creative_session(cycles=10)
        
        print("\n=== Creative Writing Scenario Complete ===")
        
    except KeyboardInterrupt:
        print("\nScenario interrupted by user")
    except Exception as e:
        print(f"Error running scenario: {e}")


if __name__ == "__main__":
    main()