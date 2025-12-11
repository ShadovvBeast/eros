# Enhanced Visual Logging Guide

## Overview

The E.R.O.S system now provides a beautiful, comprehensive visual logging system that shows you exactly what your agent is thinking, deciding, and learning in real-time. Every cycle is displayed with rich visual formatting, detailed reasoning prompts, and complete transparency into the agent's decision-making process.

## What Each Cycle Contains

Every agent cycle consists of 7 main phases:

1. **Logos Interpretation** - Reasoning and intention generation
2. **Ethos Validation** - Safety and ethics checking  
3. **Tool Execution** - Optional tool usage based on intention
4. **Pathos State Update** - Affective dynamics and reward processing
5. **Memory Storage Decision** - Salience-based memory management
6. **Attractor Dynamics** - Behavioral pattern updates
7. **Instrumentation & Logging** - Metrics and event recording

## Enhanced Log Information

### Enhanced Visual Cycle Display (Always Shown)
```
────────────────────────────────────────────────────────────────────────────────
[15:26:01.952] 🔄 CYCLE 1 COMPLETED ⚡ 0.024s
  💭 Intention: Explore current environment and assess available opportunities...
  🏷️  Category: EXPLORATION
  🎯 Rewards: Internal: +0.123 | External: +0.000 | Total: +0.123
  💾 Memory: STORED (Salience: 0.456)
  🔧 Tool Used: None
```

**Visual Elements Explained:**
- **🔄 CYCLE X COMPLETED**: Clear cycle header with visual separator
- **⚡🟢🟡🔴**: Performance indicators (⚡ = <0.01s, 🟢 = <0.1s, 🟡 = <1s, 🔴 = >1s)
- **💭 Intention**: What the agent decided to do (up to 70 chars)
- **🏷️ Category**: Semantic category in uppercase
- **🎯 Rewards**: Internal, external, and total rewards with +/- indicators
- **💾 Memory**: Storage status with salience score
- **🔧 Tool Used**: Which tool was executed (if any)

### Detailed Phase Logging (Optional)

Enable detailed logging using the "Detailed Cycle Logging" checkbox in the Log & Audit tab.

#### Agent Reasoning (Thinking Prompts)
```
[15:26:01.950] [LOGOS] [DEBUG] Cycle 1 - Agent Reasoning
    • thinking_prompt: I am currently in a balanced and contemplative state. Based on my exploration mindset and drawing from 3 recent experiences, I am considering: 'Explore current environment and assess available opportunities with balanced attention'. This aligns with my current preference for exploration activities.
    • semantic_category: exploration
    • interest_signal: 0.789
    • recent_memories_count: 3
    • intention_preview: Explore current environment and assess available opportunities...
```

#### Semantic Interpretation
```
[15:26:01.951] [LOGOS] [DEBUG] Cycle 1 - Semantic interpretation
    • semantic_category: exploration
    • state_magnitude: 0.567
    • memory_themes: ['curiosity', 'discovery', 'learning']
    • intention_preview: Explore current environment and assess available opportunities
```

#### Tool Decision & Execution
```
[15:26:02.154] [LOGOS] [DEBUG] Cycle 2 - Tool Decision
    • tool_reasoning: To accomplish 'Analyze gathered information to identify patterns and insights', I have determined that the 'analyze_data' tool is most appropriate. This tool will help me process and understand available information which directly supports my current analysis objective.
    • selected_tool: analyze_data
    • tool_args: {'data_source': 'recent_observations', 'analysis_type': 'pattern_detection'}

[15:26:02.155] [TOOLS] [DEBUG] Cycle 2 - Executing tool
    • tool_name: analyze_data
    • validation_status: APPROVED
    • execution_context: For Analyze gathered information to identify patterns...

[15:26:02.156] [TOOLS] [DEBUG] Cycle 2 - Tool result
    • tool_name: analyze_data
    • result_analysis: SUCCESS - Tool achieved intended outcome
    • external_reward: +1.000
    • impact: Positive reinforcement
```

#### Pathos Update Phase
```
[15:26:01.953] [PATHOS] [DEBUG] Cycle 1 - Pathos update
internal_reward: 0.123 | external_reward: 1.000 | total_reward: 1.123 | 
state_change: 0.045 | retrieved_memories: 3
```

#### Memory Storage Phase
```
[15:26:01.954] [MEMORY] [DEBUG] Cycle 1 - Memory stored
salience: 0.456 | total_traces: 15 | total_reward: 1.123

# OR if not stored:
[15:26:01.954] [MEMORY] [DEBUG] Cycle 1 - Memory not stored
salience: 0.123 | threshold: below_threshold
```

## Enhanced Log Tab Features

### Visual Enhancements
- **🎨 Color-coded components**: LOGOS (Blue), PATHOS (Red), MEMORY (Orange), TOOLS (Green), etc.
- **📊 Performance indicators**: Visual speed indicators (⚡🟢🟡🔴) based on cycle duration
- **🎯 Reward visualization**: Icons and +/- formatting for clear reward tracking
- **💾 Memory status**: Clear storage indicators with salience scores
- **─── Visual separators**: Clean separation between cycles for easy reading
- **📋 Structured debug info**: Bullet-pointed details for easy scanning

### Interactive Controls
- **Level Filter**: ALL, ERROR, WARNING, INFO, DEBUG
- **Component Filter**: ALL, SESSION, AGENT, PATHOS, MEMORY, LOGOS, TOOLS, SYSTEM
- **Search**: Text search across log messages and details
- **☑️ Detailed Cycle Logging**: Toggle for complete reasoning transparency
- **Auto-scroll**: Automatic scrolling to latest entries

### Statistics Display
- Total log entries
- Error/Warning/Info counts  
- Last update timestamp

### Export Options
- Export current log to text file
- Export filtered log entries
- Export log statistics to JSON
- Copy log to clipboard

## New Features: Agent Reasoning Transparency

### Thinking Prompts
The system now shows what the agent is "thinking" about:
- Current emotional/energy state analysis
- Memory context consideration
- Intention reasoning process
- Preference alignment explanation

### Tool Decision Reasoning
See exactly why the agent chooses specific tools:
- Goal-to-tool mapping explanation
- Tool capability matching
- Decision validation process
- Success/failure impact analysis

### Visual Formatting
- **Enhanced readability** with icons and colors
- **Structured information** with bullet points
- **Performance visualization** with speed indicators
- **Clear cycle separation** with visual dividers

## Usage Tips

1. **Start with enhanced cycle view** to see beautiful visual summaries
2. **Enable "Detailed Cycle Logging"** to see complete reasoning process
3. **Use component filters** to focus on specific system parts (LOGOS for reasoning, TOOLS for actions)
4. **Search for specific intentions** or "thinking_prompt" to track agent thoughts
5. **Watch performance indicators** to see cycle efficiency in real-time
6. **Export logs** for offline analysis or sharing with enhanced formatting

## Performance Impact

- Basic cycle logging: Minimal overhead
- Detailed phase logging: ~5-10% performance impact
- Log display updates: Real-time with 1-second refresh

## Troubleshooting

If you don't see detailed information:
1. Ensure "Detailed Cycle Logging" is checked
2. Set Level Filter to "ALL" or "DEBUG"  
3. Check that the agent session is running
4. Verify Component Filter includes "AGENT" or "ALL"

The enhanced logging system provides complete transparency into E.R.O.S operations while maintaining excellent performance for production use.