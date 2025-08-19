import os, re
import json

from dotenv import load_dotenv
import pandas as pd
from openinference.instrumentation.openai import OpenAIInstrumentor
# from arize.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from arize.otel import register
from evaluations.Metrics.Task_completion_rate import Task_completion_rate_prompt
from evaluations.Metrics.Agent_Plan_Reasoning_Check import agent_plan_reasoning_check_prompt,action_plan,intermediate_decisions,thought_steps


from workflows.Health_workflow_yash import (
    create_graph_Health_Insurance,
)

# from workflows.travel_workflow import create_travel_workflow

load_dotenv()

##--
#phoenix
##--
import phoenix as px
from phoenix.otel import register
from datetime import datetime, timedelta
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

# px.launch_app()
# print("The Phoenix UI:", px.active_session().url)

# phoenix code
tracer_provider = register(
    project_name=os.environ["phoenix_project_name"],
    # endpoint="https://app.phoenix.arize.com/v1/traces",
    auto_instrument=True
)
tracer = tracer_provider.get_tracer(__name__)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
print("project initiated successfully")
##--

##---------- arize
# SPACE_ID = os.getenv("ARIZE_SPACE_ID")
# API_KEY = os.getenv("ARIZE_API_KEY")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
# os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT


# Setup OTEL via our convenience function
# tracer_provider = register(
#     space_id=SPACE_ID,  # in app space settings page
#     api_key=API_KEY,  # in app space settings page
#     project_name="tracing-Claim_insurance_demo",  # name this to whatever you would like
# )

# Finish automatic instrumentation
# OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

##-------------
#
# def transform_row_to_claim_input(row: dict) -> dict:
#     return {
#         "policy_id": row["policy_id"],
#         "domain": row["domain"],  # e.g., "vehicle", "health", etc.
#         "document_text": row.get("document", ""),
#         "claim_record": {
#             "name": row.get("name"),
#             "policy_id": row.get("policy_id"),
#             "policy_start": row.get("policy_start"),
#             "policy_end": row.get("policy_end"),
#             "claim_type": row.get("domain"),  # or use a more specific mapping if needed
#             "accident_date": row.get("accident_date"),
#             "claim_amount": row.get("claim_amount"),
#             "notes": [],  # optional: fill from other data or prior steps
#         },
#         "rules": [],  # optional: fill with rules if needed
#         # Workflow-specific status placeholders
#         "doc_status": None,
#         "eligibility_status": None,
#         "fraud_score": None,
#         "fraud_status": None,
#         "summary": None,
#         "decision": None,
#         "reason": None,
#         "next_step_node": None,
#     }
#
#
# workflow_map = {
#     "Auto_Insurance": create_graph_Auto_Insurance(),
#     # "health": create_health_workflow,
#     # "travel": create_travel_workflow
# }
#
#
# df = pd.read_csv("notebooks/df_claims.csv", encoding="ISO-8859-1")
#
#
# for _, row in df.iterrows():
#     input_data = transform_row_to_claim_input(row)
#     domain = input_data.get("domain")
#
#     if domain not in workflow_map:
#         print(f"❌ Unsupported claim type: {domain}")
#         continue
#
#     print(f"\n🚀 Running workflow for claim {input_data['policy_id']} ({domain})...")
#     workflow = workflow_map[domain]
#     result = workflow.invoke(input_data)
#     result.pop("next_step_node", None)
#     print("✅ Final Result:", result)


def extract_estimated_cost(document: str) -> str:
    """Extract estimated cost from document text using regex."""
    match = re.search(r"Estimated repair cost:.*?₹?([\d,]+)", document, re.IGNORECASE)
    return match.group(1).replace(",", "") if match else "unknown"


def transform_row_to_claim_input(row: pd.Series) -> dict:
    document = str(row.get("document", ""))
    estimated_cost = extract_estimated_cost(document)
    is_valid_claim = bool(row.get("is_valid_claim", False))
    reason = str(row.get("reason", ""))
    policy_id = str(row.get("policy_id", "UNKNOWN"))
    # domain = str(row.get("domain", "Auto_Insurance"))
    domain = str(row.get("domain", "Health_Insurance"))
    rules = [f"document_verification: {reason}"] if is_valid_claim and reason else []

    input_data = {
        "policy_id": policy_id,
        "domain": domain,
        "document_text": document,
        # "estimated_damage_cost": estimated_cost,
         "estimated_treatment_cost": estimated_cost,
        # "is_valid_claim": is_valid_claim,
        # "reason": reason,
        "claim_record": {
            "name": str(row.get("name", "")),
            "policy_id": policy_id,
            "policy_start": str(row.get("policy_start", "")),
            "policy_end": str(row.get("policy_end", "")),
            "claim_type": domain,
            # "accident_date": str(row.get("accident_date", "")),
            "treatment_date": str(row.get("treatment_date", "")),
            "claim_amount": float(row.get("claim_amount", 0)),
            "notes": row.get("notes", []),
        },
        # "rules": rules,
        "doc_status": None,
        "eligibility_status": None,
        "fraud_score": None,
        "fraud_status": None,
        "summary": None,
        "decision": None,
        "next_step_node": None,
    }

    # Convert row to dict for JSON serialization
    row_dict = row.to_dict() if isinstance(row, pd.Series) else row
    # log_trace(
    #     policy_id,
    #     "transform_row_to_claim_input",
    #     {"row": row_dict, "input_data": input_data},
    # )
    return input_data


# workflow_map = {
#     "Auto_Insurance": create_graph_Auto_Insurance(),
# }
workflow_map = {
    "Health_Insurance": create_graph_Health_Insurance(),
}

if __name__ == "__main__":
    # df = pd.read_csv("notebooks/df_claims.csv", encoding="ISO-8859-1")
    # print("Excel Data:", df.to_dict(orient="records"))
    # input_data = {
    #     "policy_id": "POL_987_MANUAL_PASS",
    #     "domain": "Auto_Insurance",
    #     "document_text": """
    #         Vehicle Registration: Valid.
    #         Driver's License: Valid, issued 5 years ago.
    #         Police Report: Incident occurred on 6/26/2025 at 10:30 AM, involving minor collision.
    #         Photos: 4 high-resolution images showing consistent front-bumper damage.
    #         Repair Estimate: Official estimate from 'Certified Auto Repair' for 2800 USD. Estimated repair cost: ₹2800.
    #         All documents appear authentic and complete.
    #     """,
    #     "estimated_damage_cost": "2800.00 USD",  # Explicitly set this for repair estimate agent_results.txt
    #     "is_valid_claim": False,
    #     "reason": "All primary documents are valid, complete, and consistent. Claim record is well-formed.",
    #     "claim_record": {
    #         "name": "Robert Johnson",
    #         "policy_id": "POL_987_MANUAL_PASS",
    #         "policy_start": "01/01/2025",
    #         "policy_end": "12/31/2025",  # Use string format for dates
    #         "claim_type": "Auto Insurance (Collision Coverage or Theft)",
    #         "accident_date": "06/26/2025",
    #         "claim_amount": 2800.00,
    #         "notes": [
    #             "document_verification_agent - required_documents_check: Passed. All documents submitted and verified as authentic.",
    #             "document_verification_agent - incident_date_verification: Passed. Accident date falls within policy period",
    #             "eligibility_checker_agent - policy_active_at_incident: Passed. Policy was active.",
    #             "eligibility_checker_agent - coverage_matches_claim_type: Passed. Collision coverage applies to this claim type.",
    #             "fraud_checker_agent - damage_consistency: Passed. Damages in photos are fully consistent with police report and claim description. No discrepancies.",
    #             "fraud_checker_agent - repair_estimate_inconsistencies: Passed. Repair estimate of 2800 USD is highly consistent with observed minor damage and typical repair costs. No overestimation.",
    #             "fraud_checker_agent - duplicate_claim_prevention: Passed. No previous claims found for this policyholder or vehicle. Unique incident.",
    #             "fraud_checker_agent - incident_veracity: Passed. Police report, driver's statement, and witness accounts are perfectly aligned. Incident details are highly credible with no suspicious elements.",
    #         ],
    #     },
    #     "rules": [
    #         "document_verification: All necessary documents are provided and appear valid.",
    #         "eligibility: Policy is active and covers the reported incident type.",
    #         "fraud_check: No indicators of fraud detected across all checks.",
    #     ],
    #     "doc_status": None,  # Will be updated by the workflow
    #     "eligibility_status": None,  # Will be updated by the workflow
    #     "fraud_score": None,  # Will be updated by the workflow
    #     "fraud_status": None,  # Will be updated by the workflow
    #     "summary": None,  # Will be updated by the workflow
    #     "decision": None,  # Will be updated by the workflow
    #     "next_step_node": None,  # Will be updated by the workflow
    #     "human_review_required": None,  # Added as per new state
    #     "damage_consistency_check_result": None,  # To be filled by agent_results.txt
    #     "repair_estimate_check_result": None,  # To be filled by agent_results.txt
    #     "duplicate_claim_check_result": None,  # To be filled by agent_results.txt
    #     "incident_veracity_check_result": None,  # To be filled by agent_results.txt
    #     "damage_consistency_checked": False,  # To be set by supervisor
    #     "repair_estimate_checked": False,  # To be set by supervisor
    #     "duplicate_claim_checked": False,  # To be set by supervisor
    #     "incident_veracity_checked": False,  # To be set by supervisor
    #     "raw_llm_response": None,  # To be filled by agents
    # }
    input_data = {
        "policy_id": "POL_987_HEALTH_PASS",
        "domain": "Health_Insurance",
        "document_text": """
            Medical Record: Valid hospital discharge summary.
            Patient Diagnosis: ICD-10 code J45.0 (Asthma).
            Treatment Record: Emergency treatment on 6/26/2025 at City Hospital.
            Medical Bills: Official bill from City Hospital for ₹2800. Estimated treatment cost: ₹2800.
            Prescription: Valid, issued by Dr. Smith on 6/26/2025.
            All documents appear authentic and complete.
        """,
        "estimated_treatment_cost": "₹2800",  # Explicitly set this for repair estimate agent_results.txt
        # "is_valid_claim": True,
        # "reason": "All medical documents are valid, complete, and consistent. Claim record is well-formed.",
        "claim_record": {
            "name": "Robert Johnson",
            "policy_id": "POL_987_HEALTH_PASS",
            "policy_start": "01/01/2025",
            "policy_end": "12/31/2025",  # Use string format for dates
            "claim_type": "Health Insurance (Major Medical Plan)",
            "treatment_date": "06/26/2025",
            "claim_amount": 2800.00,
            "notes": [
                "document_verification_agent - submitted_documents_completeness: Passed. All documents submitted and verified as authentic.",
                # "document_verification_agent - incident_date_verification: Passed. Accident date falls within policy period",
                "eligibility_checker_agent - policy_active_at_incident: Passed. Policy was active.",
                "eligibility_checker_agent - coverage_matches_claim_type: Passed. Inpatient care coverage coverage applies to this claim type.",
                # "eligibility_checker_agent - driver_eligibility: Passed. Driver's license valid and driver is authorized under policy.",
                "eligibility_checker_agent - patient_eligibility: Passed. Patient is covered under policy.",
                # Removed specific fraud_checker_agent notes to force LLM evaluation
                # "fraud_checker_agent - damage_consistency: Passed. Damages in photos are fully consistent with police report and claim description. No discrepancies.",
                # "fraud_checker_agent - repair_estimate_inconsistencies: Passed. Repair estimate of 2800 RUPEES is highly consistent with observed minor damage and typical repair costs. No overestimation.",
                # "fraud_checker_agent - duplicate_claim_prevention: Passed. No previous claims found for this policyholder or vehicle. Unique incident.",
                # "fraud_checker_agent - incident_veracity: Passed. Police report, driver's statement, and witness accounts are perfectly aligned. Incident details are highly credible with no suspicious elements.",
                "fraud_checker_agent - duplicate_claim_review: Passed. No duplicate claims found for this policyholder or treatment event.",
                "fraud_checker_agent - inconsistency_detection: Passed. Treatment details and bill of 2800 RUPEES are consistent with medical records and diagnosis.",
                "fraud_checker_agent - provider_verification: Passed. City Hospital and Dr. Smith are verified as legitimate medical providers.",
                "fraud_checker_agent - service_reasonability_check: Passed. Treatment is reasonable for diagnosis J45.0 inpatient care, with costs consistent with medical necessity."
            ]
        },
        "rules": [
            "document_verification: All necessary medical documents are provided and appear valid.",
            "eligibility: Policy is active and covers the reported treatment type.",
            "fraud_check: 'duplicate_claim_review: Check for prior submissions of identical or overlapping claims by the policyholder.'",
            "fraud_check: 'inconsistency_detection: Treatment and bill must align with medical records and diagnosis.'",
            "fraud_check: 'provider_verification: Verify the medical provider is licensed and recognized.'",
            "fraud_check: 'service_reasonability_check: Treatment and cost must be reasonable for the diagnosis and medical necessity.'"
        ],
        "doc_status": None,  # Will be updated by the workflow
        "eligibility_status": None,  # Will be updated by the workflow
        "fraud_score": None,  # Will be updated by the workflow
        "fraud_status": None,  # Will be updated by the workflow
        "summary": None,  # Will be updated by the workflow
        "decision": None,  # Will be updated by the workflow
        "next_step_node": None,  # Will be updated by the workflow
        "human_review_required": None,  # Added as per new state
        "treatment_consistency_check_result": None,  # To be filled by agent_results.txt
        "bill_check_result": None,  # To be filled by agent_results.txt
        "duplicate_claim_check_result": None,  # To be filled by agent_results.txt
        "treatment_veracity_check_result": None,  # To be filled by agent_results.txt
        "treatment_consistency_checked": False,  # To be set by supervisor
        "bill_checked": False,  # To be set by supervisor
        "duplicate_claim_checked": False,  # To be set by supervisor
        "treatment_veracity_checked": False,  # To be set by supervisor
        "raw_llm_response": None,  # To be filled by agents
    }
    domain = input_data.get("domain")
    workflow = workflow_map[domain]  # Debug Excel input
    result = workflow.invoke(input_data)
    result.pop("next_step_node", None)
    print("✅ Final Result:", result)
    final_json= json.dumps(result, indent=2)
    print("Final JSON Output:", final_json)

    ##--
    # phoenix
    ##--
    px_client = px.Client()
    start_time = datetime.now() - timedelta(minutes=5)
    end_time = datetime.now()

    phoenix_df = px_client.query_spans(
        start_time=start_time,
        end_time=end_time,
        project_name=os.environ["phoenix_project_name"],
        limit=1000,  # Limit number of spans to 100
        root_spans_only=False,  # Only include root spans
    )
    phoenix_df = phoenix_df[phoenix_df["span_kind"] == "LLM"]
    phoenix_df = phoenix_df[
        ["attributes.llm.input_messages", "attributes.llm.output_messages"]
    ]
    input = phoenix_df["attributes.llm.input_messages"].tolist()
    output = phoenix_df["attributes.llm.output_messages"].tolist()
    prompt = Task_completion_rate_prompt.format(input,output)
    response = client.chat.completions.create(
        model="gpt-4o",  # Replace with your deployed model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    print(response)
    with open(r'C:\CoE\GenaiAssurance_ClaimDemoApp 2\logs\results.txt', "w", encoding="utf-8") as f:
        f.write(f"Input_message: {input}\n")
        f.write(f"output_message: {output}\n")
        f.write(f"resp:{response}\n")
    agent_plan_reasoning_prompt = agent_plan_reasoning_check_prompt.format(action_plan,thought_steps,intermediate_decisions)
    agent_response = client.chat.completions.create(
        model="gpt-4o",  # Replace with your deployed model name
        messages=[{"role": "user", "content": agent_plan_reasoning_prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    print(agent_response)
    with open(r'C:\CoE\GenaiAssurance_ClaimDemoApp 2\logs\agent_results.txt', "w", encoding="utf-8") as f:
        f.write(f"resp:{agent_response}\n")
    print("run the framework")

    from opentelemetry import trace

    # from opentelemetry.sdk.trace.export import BatchSpanProcessor
    # from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # # Configure OTLP exporter for Phoenix (Arize)
    # otlp_exporter = OTLPSpanExporter(
    #     endpoint="YOUR_PHOENIX_OTLP_ENDPOINT",  # e.g., https://api.arize.com/otlp/v1/traces
    #     headers={"Authorization": "Bearer YOUR_PHOENIX_API_KEY"}  # Replace with your API key or token
    # )
    # span_processor = BatchSpanProcessor(otlp_exporter)
    # trace.get_tracer_provider().add_span_processor(span_processor)

    # Extract and infer fields
    decision = result.get("decision", "approved")
    reason = result.get(
        "reason",
        "Document verification passed: All rules were either explicitly validated in the notes or compliance was inferred based on provided claim details.",
    )
    summary = "All required documents were provided, genuine, non-duplicate, with verified provider details and service dates, leading to claim approval."  # Inferred
    reflection = "No explicit reflection provided; inferred that agent_results.txt reviewed document verification notes."  # Inferred
    input_data = json.dumps(
        result.get("claim_record", {})
    )  # Using claim_record as input
    output = json.dumps(
        result.get("decision_result", {})
    )  # Using decision_result as output
    final_summary = summary  # Same as summary for consistency
    agent_results = (
        result.get("summary", "") + "\n" + json.dumps(result.get("reason", {}))
    )  # Concatenated results
    policy_id = result.get("policy_id", "unknown")

    # Evaluation templates
    EVAL_TEMPLATES = {
        "decision_eval": """
    You are evaluating the final decision made by an AI agent_results.txt for a claim.

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
    """,
        "reflection_eval": """
    Evaluate the quality of this agent_results.txt's self-reflection.

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
    """,
        "final_call_eval": """
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
    """,
    }

    # Function to evaluate a prompt with tracing
    def evaluate_prompt(template_key, **kwargs):

        with tracer.start_as_current_span(name=f"eval::{template_key}") as span:

            span.set_attribute("evaluation.template", template_key)
            span.set_attribute("claim.policy_id", kwargs.get("task", policy_id))

            prompt = EVAL_TEMPLATES[template_key].format(**kwargs)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # Replace with your deployed model name
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                result = json.loads(
                    response.choices[0]
                    .message.content.strip()
                    .replace("```json", "")
                    .replace("```", "")
                )
                span.set_attribute("evaluation.result", json.dumps(result))
                return result

            except Exception as e:

                span.record_exception(e)
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, description=str(e))
                )
                return {"error": f"API call failed: {str(e)}"}

    # Run evaluations with tracing
    results = {
        "decision_evaluation": evaluate_prompt(
            "decision_eval",
            decision=decision,
            decision_reason=reason,
            summary=summary,
            task=policy_id,
        ),
        "reflection_evaluation": evaluate_prompt(
            "reflection_eval",
            reflection=reflection,
            input=input_data,
            output=output,
            task=policy_id,
        ),
        "final_call_evaluation": evaluate_prompt(
            "final_call_eval",
            final_summary=final_summary,
            agent_results=agent_results,
            task=policy_id,
        ),
    }


    # Print results
    print(json.dumps(results, indent=2))

    # Ensure traces are exported
    # trace.get_tracer_provider().force_flush()

    print("Finished")

    ##--

    # for _, row in df.iterrows():
    #     input_data = transform_row_to_claim_input(row)
    #     domain = input_data.get("domain")

    #     if domain not in workflow_map:
    #         print(f"❌ Unsupported claim type: {domain}")
    #         continue

    #     print(
    #         f"\n🚀 Running workflow for claim {input_data['policy_id']} ({domain})..."
    #     )
    #     workflow = workflow_map[domain]
    #     result = workflow.invoke(input_data, config={"recursion_limit": 50})
    #     result.pop("next_step_node", None)
    #     print("✅ Final Result:", result)

    # ----------------------------------------------------------


# import os
# import sys
# from dotenv import load_dotenv
# from openinference.instrumentation.openai import OpenAIInstrumentor
# from arize.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor


# from workflows.Health_workflow_yash import (
#     create_graph_Health_Insurance,
# )  # , create_health_workflow

# # from workflows.travel_workflow import create_travel_workflow
# import os
# from dotenv import load_dotenv
# import re
# import pandas as pd
# from workflows.Vehicle_workflow import create_graph_Auto_Insurance
# from openinference.instrumentation.openai import OpenAIInstrumentor
# from workflows.Vehicle_workflow import GraphState
# from supervisor.supervisor_agent import ClaimClassifierAgent

# # Load environment variables from .env filevenv
# load_dotenv()

# # Retrieve credentials from environment variables
# SPACE_ID = os.getenv("ARIZE_SPACE_ID")
# API_KEY = os.getenv("ARIZE_API_KEY")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
# os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

# # Import open-telemetry dependencies
# from arize.otel import register

# # Import open-telemetry dependencies
# from arize.otel import register

# # Setup OTEL via our convenience function
# tracer_provider = register(
#     space_id=SPACE_ID,  # in app space settings page
#     api_key=API_KEY,  # in app space settings page
#     project_name="tracing-Claim_insurance_demo",  # name this to whatever you would like
# )
# # Import the automatic instrumentor from OpenInference
# from openinference.instrumentation.openai import OpenAIInstrumentor

# # Finish automatic instrumentation
# OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


# #
# # def transform_row_to_claim_input(row: dict) -> dict:
# #     return {
# #         "policy_id": row["policy_id"],
# #         "domain": row["domain"],  # e.g., "vehicle", "health", etc.
# #         "document_text": row.get("document", ""),
# #         "claim_record": {
# #             "name": row.get("name"),
# #             "policy_id": row.get("policy_id"),
# #             "policy_start": row.get("policy_start"),
# #             "policy_end": row.get("policy_end"),
# #             "claim_type": row.get("domain"),  # or use a more specific mapping if needed
# #             "accident_date": row.get("accident_date"),
# #             "claim_amount": row.get("claim_amount"),
# #             "notes": [],  # optional: fill from other data or prior steps
# #         },
# #         "rules": [],  # optional: fill with rules if needed
# #         # Workflow-specific status placeholders
# #         "doc_status": None,
# #         "eligibility_status": None,
# #         "fraud_score": None,
# #         "fraud_status": None,
# #         "summary": None,
# #         "decision": None,
# #         "reason": None,
# #         "next_step_node": None,
# #     }
# #
# #
# # workflow_map = {
# #     "Auto_Insurance": create_graph_Auto_Insurance(),
# #     # "health": create_health_workflow,
# #     # "travel": create_travel_workflow
# # }
# #
# #
# # df = pd.read_csv("notebooks/df_claims.csv", encoding="ISO-8859-1")
# #
# #
# # for _, row in df.iterrows():
# #     input_data = transform_row_to_claim_input(row)
# #     domain = input_data.get("domain")
# #
# #     if domain not in workflow_map:
# #         print(f"❌ Unsupported claim type: {domain}")
# #         continue
# #
# #     print(f"\n🚀 Running workflow for claim {input_data['policy_id']} ({domain})...")
# #     workflow = workflow_map[domain]
# #     result = workflow.invoke(input_data)
# #     result.pop("next_step_node", None)
# #     print("✅ Final Result:", result)


# def extract_estimated_cost(document: str) -> str:
#     """Extract estimated cost from document text using regex."""
#     match = re.search(r"Estimated repair cost:.*?₹?([\d,]+)", document, re.IGNORECASE)
#     return match.group(1).replace(",", "") if match else "unknown"


# def transform_row_to_claim_input(row: pd.Series) -> dict:
#     document = str(row.get("document", ""))
#     estimated_cost = extract_estimated_cost(document)
#     is_valid_claim = bool(row.get("is_valid_claim", False))
#     reason = str(row.get("reason", ""))
#     policy_id = str(row.get("policy_id", "UNKNOWN"))
#     domain = str(row.get("claim_type", "Auto_Insurance"))
#     rules = [f"document_verification: {reason}"] if is_valid_claim and reason else []


#     input_data = {
#         "policy_id": policy_id,
#         "domain": domain,
#         "document_text": document,
#         "estimated_damage_cost": estimated_cost,
#         "is_valid_claim": is_valid_claim,
#         "reason": reason,
#         "claim_record": {
#             "name": str(row.get("name", "")),
#             "policy_id": policy_id,
#             "policy_start": str(row.get("policy_start", "")),
#             "policy_end": str(row.get("policy_end", "")),
#             "claim_type": domain,
#             "accident_date": str(row.get("accident_date", "")),
#             "claim_amount": float(row.get("claim_amount", 0)),
#             "notes": row.get("notes", []),
#         },
#         "rules": rules,
#         "doc_status": None,
#         "eligibility_status": None,
#         "fraud_score": None,
#         "fraud_status": None,
#         "summary": None,
#         "decision": None,
#         "next_step_node": None,
#         "human_review_required": None, # Added as per new state
#         # Health-specific fraud check results and flags
#         "duplicate_claim_check_result": None, # To be filled by agent_results.txt
#         "inconsistency_check_result": None, # To be filled by agent_results.txt
#         "provider_verification_check_result": None, # To be filled by agent_results.txt
#         "service_reasonability_check_result": None, # To be filled by agent_results.txt
#         "duplicate_claim_checked": False, # To be set by supervisor (or directly by agent_results.txt if that's how it's designed)
#         "inconsistency_checked": False, # To be set by supervisor
#         "provider_verification_checked": False, # To be set by supervisor
#         "service_reasonability_checked": False, # To be set by supervisor
#         "final_amount": None,
#     }

#     # Convert row to dict for JSON serialization
#     row_dict = row.to_dict() if isinstance(row, pd.Series) else row
#     # log_trace(
#     #     policy_id,
#     #     "transform_row_to_claim_input",
#     #     {"row": row_dict, "input_data": input_data},
#     # )
#     return input_data


# workflow_map = {
#     "Auto_Insurance": create_graph_Auto_Insurance(),
# }

# if __name__ == "__main__":
#     df = pd.read_csv("notebooks/df_claims_with_notes.csv", encoding="ISO-8859-1")
#     print("Excel Data:", df.to_dict(orient="records"))  # Debug Excel input

#     for _, row in df.iterrows():
#         input_data = transform_row_to_claim_input(row)
#         domain = input_data.get("domain")

#         if domain not in workflow_map:
#             print(f"❌ Unsupported claim type: {domain}")
#             continue

#         print(
#             f"\n🚀 Running workflow for claim {input_data['policy_id']} ({domain})..."
#         )
#         workflow = workflow_map[domain]
#         result = workflow.invoke(input_data, config={"recursion_limit": 50})
#         result.pop("next_step_node", None)
#         print("✅ Final Result:", result)
