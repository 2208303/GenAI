# evaluation/prompts.py

TOOL_SELECTION_EVAL_TEMPLATE = """
You are an evaluator assessing whether the AI selected the correct tool for a step in the insurance claim process.

Tool Invoked: {tool_invoked}
Available Tools: {available_tools}
Step Input: {step_input}
Ground Truth Tool: {ground_truth_tool}

Was the tool selection appropriate?

Respond in JSON:
{{
  "score": <float between 0.0 and 1.0>,
  "justification": "<brief explanation>",
  "tool_correct": <true/false>
}}
"""


TOOL_CALL_EVAL_TEMPLATE = """
You are evaluating the correctness of a tool (agent) call in an insurance claim processing system.

Tool Name: {tool_name}
Inputs Provided:
{tool_inputs}

Output Produced:
{tool_output}

Does the tool have all necessary inputs to perform its task? Was the output consistent with those inputs and the tool's role?

Respond in JSON:
{{
  "score": <0.0 to 1.0>,
  "input_completeness": "<brief comment>",
  "output_validity": "<brief comment>",
  "tool_call_correct": <true/false>
}}
"""

DECISION_EVAL_TEMPLATE = """
You are evaluating the final decision made by an AI agent for a claim.

Decision: {decision}
Reason: {decision_reason}
Summary: {summary}

Was this decision justified?

Respond in JSON:
{{
  "score": <0.0 to 1.0>,
  "decision_justified": <true/false>,
  "justification": "<explanation>"
}}
"""

REFLECTION_EVAL_TEMPLATE = """
Evaluate the quality of this agent's self-reflection.

Reflection: {reflection}
Input: {input}
Output: {output}

Was the reflection thoughtful and accurate?

Respond in JSON:
{{
  "score": <0.0 to 1.0>,
  "reflection_quality": "<brief comment>",
  "reflection_valid": <true/false>
}}
"""


FINAL_CALL_EVAL_TEMPLATE = """
You are reviewing the overall insurance claim workflow.

Final Summary: {final_summary}
Agent Results: {agent_results}

Was the process consistent and complete?

Respond in JSON:
{{
  "overall_score": <0.0 to 1.0>,
  "consistent": <true/false>,
  "comment": "<brief final assessment>"
}}
"""
