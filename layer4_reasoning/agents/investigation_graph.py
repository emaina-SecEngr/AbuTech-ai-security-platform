"""
Layer 4 — LLM Reasoning
LangGraph Investigation Orchestration

This module wires all five specialist agents
into a directed graph using LangGraph.

Graph Structure:
    START
      ↓
    TriageAgent
      ↓ (conditional)
    ┌─────────────────────┐
    │ INVESTIGATE         │ MONITOR/CLOSE
    ↓                     ↓
    IntelAgent         ReportAgent
      ↓                     ↓
    InvestigationAgent    END
      ↓ (conditional)
    ┌─────────────────────┐
    │ COMPROMISE          │ NO COMPROMISE
    ↓                     ↓
    ResponseAgent      ReportAgent
      ↓                     ↓
    ReportAgent           END
      ↓
    END

Conditional Edges:
    After Triage:
        INVESTIGATE → IntelAgent
        MONITOR     → ReportAgent (brief)
        CLOSE       → END

    After Investigation:
        compromise=True  → ResponseAgent
        compromise=False → ReportAgent

This branching logic ensures:
    - False positives are closed immediately
    - Low priority events get monitoring reports
    - Confirmed compromises get full response plans
    - Every path ends with a report for the analyst
"""

import logging
import os
from typing import Literal

from layer4_reasoning.agents.agent_state import (
    InvestigationState,
    create_initial_state
)
from layer4_reasoning.agents.specialist_agents import (
    TriageAgent,
    IntelAgent,
    InvestigationAgent,
    ResponseAgent,
    ReportAgent
)

logger = logging.getLogger(__name__)


class InvestigationGraph:
    """
    LangGraph-based multi-agent investigation system.

    Orchestrates five specialist agents through a
    conditional directed graph. Each agent reads
    and writes to shared InvestigationState.

    Usage:
        graph = InvestigationGraph()

        # Investigate a single event
        result = graph.investigate(
            routing_result=routing_result,
            ecs_event=ecs_event,
            graph_summary=graph_stats,
            threat_summary=threat_intel
        )

        print(result["final_report"])

    Two Modes:
        With LangGraph installed:
            Uses full compiled LangGraph graph
            Proper state management and streaming

        Without LangGraph (fallback):
            Sequential agent execution
            Same logic different orchestration
            Works without installing langgraph
    """

    def __init__(
        self,
        anthropic_api_key: str = None
    ):
        self.api_key = (
            anthropic_api_key or
            os.getenv("ANTHROPIC_API_KEY", "")
        )

        # Initialize all agents
        self.triage = TriageAgent(self.api_key)
        self.intel = IntelAgent(self.api_key)
        self.investigation = InvestigationAgent(
            self.api_key
        )
        self.response = ResponseAgent(self.api_key)
        self.report = ReportAgent(self.api_key)

        # Try to build LangGraph
        self.graph = self._build_graph()

        logger.info(
            f"InvestigationGraph initialized "
            f"({'LangGraph' if self.graph else 'Sequential'} mode)"
        )

    def investigate(
        self,
        routing_result,
        ecs_event,
        graph_summary: dict = None,
        threat_summary: dict = None
    ) -> InvestigationState:
        """
        Run complete investigation on a security event.

        Takes Layer 2 routing result and Layer 3
        context and runs all appropriate agents.

        Args:
            routing_result: Layer 2 RoutingResult
            ecs_event: ECSNormalized from Layer 1
            graph_summary: Layer 3 graph statistics
            threat_summary: Layer 3 threat intel

        Returns:
            Complete InvestigationState with all
            agent findings and final report
        """
        # Create initial state from layer outputs
        initial_state = create_initial_state(
            routing_result=routing_result,
            ecs_event=ecs_event,
            graph_summary=graph_summary,
            threat_summary=threat_summary
        )

        logger.info(
            f"Starting investigation: "
            f"host={initial_state['event_host']} "
            f"risk={initial_state['overall_risk_score']:.2f}"
        )

        # Run through agent graph
        if self.graph:
            result = self._run_langgraph(initial_state)
        else:
            result = self._run_sequential(initial_state)

        logger.info(
            f"Investigation complete: "
            f"severity={result.get('severity_rating')} "
            f"agents_run={len(result.get('agent_log', []))}"
        )

        return result

    def investigate_from_state(
        self,
        state: InvestigationState
    ) -> InvestigationState:
        """
        Run investigation from pre-built state.

        Used for testing and replay scenarios
        where state is already constructed.
        """
        if self.graph:
            return self._run_langgraph(state)
        else:
            return self._run_sequential(state)

    # ============================================================
    # GRAPH CONSTRUCTION
    # ============================================================

    def _build_graph(self):
        """
        Build LangGraph compiled graph.

        Returns None if LangGraph not available.
        Falls back to sequential execution.
        """
        try:
            from langgraph.graph import (
                StateGraph, END
            )

            workflow = StateGraph(InvestigationState)

            # Add agent nodes
            workflow.add_node(
                "triage",
                self.triage.run
            )
            workflow.add_node(
                "intel",
                self.intel.run
            )
            workflow.add_node(
                "investigation",
                self.investigation.run
            )
            workflow.add_node(
                "response",
                self.response.run
            )
            workflow.add_node(
                "report",
                self.report.run
            )

            # Set entry point
            workflow.set_entry_point("triage")

            # Add conditional edges after triage
            workflow.add_conditional_edges(
                "triage",
                self._route_after_triage,
                {
                    "investigate": "intel",
                    "monitor": "report",
                    "close": END
                }
            )

            # Intel always goes to investigation
            workflow.add_edge("intel", "investigation")

            # Conditional edges after investigation
            workflow.add_conditional_edges(
                "investigation",
                self._route_after_investigation,
                {
                    "respond": "response",
                    "report": "report"
                }
            )

            # Response always goes to report
            workflow.add_edge("response", "report")

            # Report goes to end
            workflow.add_edge("report", END)

            compiled = workflow.compile()
            logger.info(
                "LangGraph compiled successfully"
            )
            return compiled

        except ImportError:
            logger.info(
                "LangGraph not available. "
                "Using sequential execution."
            )
            return None
        except Exception as e:
            logger.warning(
                f"LangGraph build failed: {e}. "
                f"Using sequential execution."
            )
            return None

    # ============================================================
    # ROUTING FUNCTIONS
    # These determine conditional edge paths
    # ============================================================

    def _route_after_triage(
        self,
        state: InvestigationState
    ) -> Literal["investigate", "monitor", "close"]:
        """
        Route based on triage verdict.

        This is the primary filtering step —
        only INVESTIGATE cases proceed to
        full multi-agent analysis.
        """
        verdict = state.get("triage_verdict", "CLOSE")

        if verdict == "INVESTIGATE":
            logger.info(
                "Routing: triage → investigate"
            )
            return "investigate"
        elif verdict == "MONITOR":
            logger.info(
                "Routing: triage → monitor (report)"
            )
            return "monitor"
        else:
            logger.info(
                "Routing: triage → close"
            )
            return "close"

    def _route_after_investigation(
        self,
        state: InvestigationState
    ) -> Literal["respond", "report"]:
        """
        Route based on compromise confirmation.

        Confirmed compromises require response
        planning before reporting.
        Unconfirmed go directly to report.
        """
        compromise = state.get("compromise_confirmed")
        c2_active = state.get("c2_confirmed")

        if compromise or c2_active:
            logger.info(
                "Routing: investigation → respond"
            )
            return "respond"
        else:
            logger.info(
                "Routing: investigation → report"
            )
            return "report"

    # ============================================================
    # EXECUTION METHODS
    # ============================================================

    def _run_langgraph(
        self,
        state: InvestigationState
    ) -> InvestigationState:
        """Execute using compiled LangGraph"""
        try:
            result = self.graph.invoke(state)
            return result
        except Exception as e:
            logger.error(
                f"LangGraph execution failed: {e}. "
                f"Falling back to sequential."
            )
            return self._run_sequential(state)

    def _run_sequential(
        self,
        state: InvestigationState
    ) -> InvestigationState:
        """
        Execute agents sequentially without LangGraph.

        Implements the same conditional logic as
        the LangGraph version but without the
        graph framework dependency.

        This fallback ensures your platform works
        in any environment.
        """
        # Step 1: Triage
        state = self.triage.run(state)

        triage_verdict = state.get(
            "triage_verdict", "CLOSE"
        )

        # Step 2: Route based on triage
        if triage_verdict == "CLOSE":
            logger.info(
                "Sequential: closed after triage"
            )
            return state

        if triage_verdict == "MONITOR":
            state = self.report.run(state)
            return state

        # INVESTIGATE path
        # Step 3: Intel
        state = self.intel.run(state)

        # Step 4: Investigation
        state = self.investigation.run(state)

        # Step 5: Route based on compromise
        if (
            state.get("compromise_confirmed") or
            state.get("c2_confirmed")
        ):
            state = self.response.run(state)

        # Step 6: Report
        state = self.report.run(state)

        return state

    def get_agent_log(
        self,
        state: InvestigationState
    ) -> list:
        """Return formatted agent execution log"""
        return state.get("agent_log", [])

    def print_investigation_summary(
        self,
        state: InvestigationState
    ) -> None:
        """Print investigation results to console"""
        print(f"\n{'='*60}")
        print("INVESTIGATION COMPLETE")
        print(f"{'='*60}")
        print(
            f"Host:     {state['event_host']}"
        )
        print(
            f"Severity: {state.get('severity_rating')}"
        )
        print(
            f"Verdict:  {state.get('triage_verdict')}"
        )
        print(
            f"Actor:    "
            f"{state.get('threat_actor_identified', 'Unknown')}"
        )
        print(
            f"Agents:   {len(state.get('agent_log', []))}"
        )
        print(f"\nEXECUTIVE SUMMARY:")
        print(
            state.get(
                "executive_summary",
                "No summary generated"
            )
        )

        if state.get("final_report"):
            print(state["final_report"])