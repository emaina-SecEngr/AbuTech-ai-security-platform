"""
Layer 4 — Agent Tests

Tests verify:
1. Triage agent correctly routes alerts
2. Intel agent correctly attributes threat actors
3. Investigation agent builds correct timelines
4. Response agent generates correct actions
5. Report agent produces complete reports
6. Graph routing works correctly
7. Full pipeline integration
"""

import pytest
from unittest.mock import MagicMock

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
from layer4_reasoning.agents.investigation_graph import (
    InvestigationGraph
)


# ============================================================
# STATE BUILDERS FOR TESTING
# ============================================================

def make_high_risk_state(**kwargs) -> InvestigationState:
    """High risk malware + DGA state"""
    state = InvestigationState(
        event_id="test-001",
        event_category="process",
        event_host="WKSTN-JSMITH-01",
        event_user="CORP\\jsmith",
        event_timestamp="2024-03-29T02:17:43Z",
        overall_risk_score=0.97,
        overall_verdict="MALWARE",
        malware_risk=0.97,
        intrusion_risk=None,
        dga_risk=0.90,
        malware_indicators=[
            "Scripting engine execution: powershell.exe",
            "Suspicious parent: MSBuild.exe",
            "Encoded PowerShell detected"
        ],
        attack_techniques=[
            "T1059 Command and Scripting",
            "T1566.001 Spearphishing Attachment"
        ],
        dga_indicators=[
            "Dynamic DNS provider detected",
            "duckdns domains",
            "System process DNS request"
        ],
        graph_node_count=11,
        graph_edge_count=14,
        high_risk_entities=[
            {
                "type": "process",
                "entity": "powershell.exe",
                "risk": 0.97
            },
            {
                "type": "ip_address",
                "entity": "185.220.101.45",
                "risk": 0.85
            },
            {
                "type": "domain",
                "entity": "xjf8k2mp.duckdns.org",
                "risk": 0.90
            }
        ],
        threat_connections=[
            {
                "type": "ip_address",
                "entity": "185.220.101.45",
                "risk": 0.85
            }
        ],
        host_risk_score=0.78,
        known_threat_actor=None,
        known_malware_family=None,
        ti_enrichments=[],
        triage_verdict=None,
        triage_confidence=None,
        triage_reasoning=None,
        triage_priority=None,
        threat_actor_identified=None,
        threat_actor_confidence=None,
        campaign_identified=None,
        c2_confirmed=None,
        malware_family_confirmed=None,
        intel_summary=None,
        confirmed_techniques=[],
        attack_timeline=[],
        compromise_confirmed=None,
        initial_access_vector=None,
        lateral_movement_detected=None,
        data_exfiltration_suspected=None,
        blast_radius=[],
        investigation_summary=None,
        response_actions=[],
        containment_priority=None,
        isolation_recommended=None,
        credential_reset_recommended=None,
        response_summary=None,
        final_report=None,
        executive_summary=None,
        severity_rating=None,
        agent_log=[]
    )
    for key, value in kwargs.items():
        state[key] = value
    return state


def make_low_risk_state(**kwargs) -> InvestigationState:
    """Low risk benign state"""
    state = make_high_risk_state()
    state["overall_risk_score"] = 0.087
    state["overall_verdict"] = "BENIGN"
    state["malware_risk"] = None
    state["dga_risk"] = None
    state["malware_indicators"] = []
    state["attack_techniques"] = []
    state["dga_indicators"] = []
    state["high_risk_entities"] = []
    state["threat_connections"] = []
    state["host_risk_score"] = 0.1
    for key, value in kwargs.items():
        state[key] = value
    return state


# ============================================================
# TEST CLASS — TRIAGE AGENT
# ============================================================

class TestTriageAgent:
    """Tests for TriageAgent decision logic"""

    def setup_method(self):
        self.agent = TriageAgent(
            anthropic_api_key=""
        )

    def test_high_risk_malware_gets_investigate(self):
        """
        High risk malware detection routes to INVESTIGATE.
        Risk 0.97 with MALWARE verdict must be investigated.
        """
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["triage_verdict"] == "INVESTIGATE"

    def test_high_risk_gets_critical_priority(self):
        """Critical risk events get CRITICAL priority"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["triage_priority"] == "CRITICAL"

    def test_low_risk_gets_close_or_monitor(self):
        """Low risk benign events get CLOSE or MONITOR"""
        state = make_low_risk_state()
        result = self.agent.run(state)

        assert result["triage_verdict"] in [
            "CLOSE", "MONITOR"
        ]

    def test_triage_reasoning_populated(self):
        """Triage produces human-readable reasoning"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["triage_reasoning"] is not None
        assert len(result["triage_reasoning"]) > 10

    def test_triage_confidence_populated(self):
        """Triage confidence score populated"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["triage_confidence"] is not None
        assert 0.0 <= result["triage_confidence"] <= 1.0

    def test_agent_log_updated(self):
        """Agent log entry created by triage"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert len(result["agent_log"]) >= 1
        assert result["agent_log"][0]["agent"] == (
            "TriageAgent"
        )

    def test_medium_risk_gets_monitor(self):
        """Medium risk events get MONITOR verdict"""
        state = make_high_risk_state(
            overall_risk_score=0.5,
            overall_verdict="BENIGN",
            malware_risk=0.5
        )
        result = self.agent.run(state)

        assert result["triage_verdict"] in [
            "MONITOR", "INVESTIGATE"
        ]


# ============================================================
# TEST CLASS — INTEL AGENT
# ============================================================

class TestIntelAgent:
    """Tests for IntelAgent threat attribution"""

    def setup_method(self):
        self.agent = IntelAgent(
            anthropic_api_key=""
        )

    def test_emotet_indicators_match_ta542(self):
        """
        Emotet-like indicators attributed to TA542.
        MSBuild + encoded PowerShell + duckdns
        matches Mummy Spider profile.
        """
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["threat_actor_identified"] == "TA542"

    def test_c2_confirmed_when_malware_and_dga(self):
        """
        C2 confirmed when malware + DGA both detected.
        Active malware + DGA domain = C2 active.
        """
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["c2_confirmed"] is True

    def test_no_c2_for_low_risk(self):
        """C2 not confirmed for low risk events"""
        state = make_low_risk_state()
        result = self.agent.run(state)

        assert not result["c2_confirmed"]

    def test_intel_summary_populated(self):
        """Intel summary generated"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert result["intel_summary"] is not None
        assert len(result["intel_summary"]) > 10

    def test_attack_techniques_populated(self):
        """Confirmed techniques list populated"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        assert len(
            result["confirmed_techniques"]
        ) >= 1

    def test_agent_log_updated(self):
        """Intel agent log entry created"""
        state = make_high_risk_state()
        result = self.agent.run(state)

        agent_names = [
            e["agent"] for e in result["agent_log"]
        ]
        assert "IntelAgent" in agent_names


# ============================================================
# TEST CLASS — INVESTIGATION AGENT
# ============================================================

class TestInvestigationAgent:
    """Tests for InvestigationAgent forensic analysis"""

    def setup_method(self):
        self.agent = InvestigationAgent(
            anthropic_api_key=""
        )

    def _make_post_intel_state(self) -> InvestigationState:
        """State after intel agent has run"""
        state = make_high_risk_state()
        state["triage_verdict"] = "INVESTIGATE"
        state["threat_actor_identified"] = "TA542"
        state["c2_confirmed"] = True
        state["confirmed_techniques"] = [
            "T1566.001", "T1059.001",
            "T1127", "T1071.001"
        ]
        return state

    def test_attack_timeline_built(self):
        """Attack timeline constructed from indicators"""
        state = self._make_post_intel_state()
        result = self.agent.run(state)

        assert len(result["attack_timeline"]) >= 1

    def test_compromise_confirmed_when_c2_active(self):
        """Compromise confirmed when C2 active + malware"""
        state = self._make_post_intel_state()
        result = self.agent.run(state)

        assert result["compromise_confirmed"] is True

    def test_blast_radius_includes_host(self):
        """Blast radius includes compromised host"""
        state = self._make_post_intel_state()
        result = self.agent.run(state)

        entities = [
            e["entity"]
            for e in result["blast_radius"]
        ]
        assert "WKSTN-JSMITH-01" in entities

    def test_initial_access_vector_identified(self):
        """Initial access vector identified"""
        state = self._make_post_intel_state()
        result = self.agent.run(state)

        assert result["initial_access_vector"] is not None

    def test_investigation_summary_populated(self):
        """Investigation summary generated"""
        state = self._make_post_intel_state()
        result = self.agent.run(state)

        assert result["investigation_summary"] is not None


# ============================================================
# TEST CLASS — RESPONSE AGENT
# ============================================================

class TestResponseAgent:
    """Tests for ResponseAgent recommendations"""

    def setup_method(self):
        self.agent = ResponseAgent(
            anthropic_api_key=""
        )

    def _make_post_investigation_state(
        self
    ) -> InvestigationState:
        """State after investigation agent"""
        state = make_high_risk_state()
        state["triage_verdict"] = "INVESTIGATE"
        state["c2_confirmed"] = True
        state["compromise_confirmed"] = True
        state["blast_radius"] = [
            {
                "entity": "WKSTN-JSMITH-01",
                "type": "host",
                "status": "COMPROMISED"
            }
        ]
        return state

    def test_isolation_recommended_for_compromise(self):
        """
        Host isolation recommended when compromise
        is confirmed.
        """
        state = self._make_post_investigation_state()
        result = self.agent.run(state)

        assert result["isolation_recommended"] is True

    def test_immediate_containment_for_compromise(self):
        """Immediate containment priority for compromise"""
        state = self._make_post_investigation_state()
        result = self.agent.run(state)

        assert result["containment_priority"] == "IMMEDIATE"

    def test_response_actions_generated(self):
        """Response actions list generated"""
        state = self._make_post_investigation_state()
        result = self.agent.run(state)

        assert len(result["response_actions"]) >= 1

    def test_isolate_host_action_present(self):
        """ISOLATE_HOST action present for compromise"""
        state = self._make_post_investigation_state()
        result = self.agent.run(state)

        action_types = [
            a["action"]
            for a in result["response_actions"]
        ]
        assert "ISOLATE_HOST" in action_types

    def test_credential_reset_recommended(self):
        """Credential reset recommended for compromise"""
        state = self._make_post_investigation_state()
        result = self.agent.run(state)

        assert (
            result["credential_reset_recommended"]
            is True
        )


# ============================================================
# TEST CLASS — REPORT AGENT
# ============================================================

class TestReportAgent:
    """Tests for ReportAgent report generation"""

    def setup_method(self):
        self.agent = ReportAgent(
            anthropic_api_key=""
        )

    def _make_complete_state(self) -> InvestigationState:
        """State with all previous agents complete"""
        state = make_high_risk_state()
        state["triage_verdict"] = "INVESTIGATE"
        state["triage_priority"] = "CRITICAL"
        state["triage_reasoning"] = "High risk malware"
        state["threat_actor_identified"] = "TA542"
        state["malware_family_confirmed"] = "Emotet"
        state["c2_confirmed"] = True
        state["confirmed_techniques"] = [
            "T1566.001", "T1059.001"
        ]
        state["compromise_confirmed"] = True
        state["initial_access_vector"] = (
            "Spearphishing attachment"
        )
        state["attack_timeline"] = [
            {
                "event": "Malware execution",
                "technique": "T1059.001",
                "significance": "Execution"
            }
        ]
        state["blast_radius"] = [
            {
                "entity": "WKSTN-JSMITH-01",
                "type": "host",
                "status": "COMPROMISED"
            }
        ]
        state["response_actions"] = [
            {
                "priority": 1,
                "action": "ISOLATE_HOST",
                "target": "WKSTN-JSMITH-01",
                "reason": "Active compromise"
            }
        ]
        state["containment_priority"] = "IMMEDIATE"
        state["isolation_recommended"] = True
        return state

    def test_critical_severity_for_compromise(self):
        """Compromise → CRITICAL severity"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert result["severity_rating"] == "CRITICAL"

    def test_executive_summary_generated(self):
        """Executive summary generated"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert result["executive_summary"] is not None
        assert len(result["executive_summary"]) > 10

    def test_final_report_generated(self):
        """Full investigation report generated"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert result["final_report"] is not None
        assert len(result["final_report"]) > 100

    def test_report_contains_host(self):
        """Report mentions the compromised host"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert "WKSTN-JSMITH-01" in (
            result["final_report"]
        )

    def test_report_contains_threat_actor(self):
        """Report mentions threat actor"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert "TA542" in result["final_report"]

    def test_report_contains_attck_techniques(self):
        """Report includes ATT&CK techniques"""
        state = self._make_complete_state()
        result = self.agent.run(state)

        assert "T1566.001" in result["final_report"]


# ============================================================
# TEST CLASS — INVESTIGATION GRAPH
# ============================================================

class TestInvestigationGraph:
    """Tests for full graph orchestration"""

    def setup_method(self):
        self.graph = InvestigationGraph(
            anthropic_api_key=""
        )

    def test_graph_routing_investigate_path(self):
        """
        High risk event goes through full
        investigation path.
        All agents run for confirmed compromise.
        """
        state = make_high_risk_state()
        result = self.graph.investigate_from_state(state)

        assert result["triage_verdict"] == "INVESTIGATE"
        assert result["final_report"] is not None

    def test_graph_routing_close_path(self):
        """
        Low risk event closed after triage.
        No intel or investigation agents run.
        """
        state = make_low_risk_state()
        result = self.graph.investigate_from_state(state)

        assert result["triage_verdict"] in [
            "CLOSE", "MONITOR"
        ]

    def test_full_pipeline_produces_report(self):
        """Full pipeline always produces a report"""
        state = make_high_risk_state()
        result = self.graph.investigate_from_state(state)

        assert result["final_report"] is not None

    def test_agent_log_tracks_all_agents(self):
        """Agent log tracks which agents ran"""
        state = make_high_risk_state()
        result = self.graph.investigate_from_state(state)

        agents_run = [
            e["agent"]
            for e in result["agent_log"]
        ]
        assert "TriageAgent" in agents_run
        assert "ReportAgent" in agents_run

    def test_severity_rating_populated(self):
        """Severity rating always populated"""
        state = make_high_risk_state()
        result = self.graph.investigate_from_state(state)

        assert result["severity_rating"] in [
            "CRITICAL", "HIGH", "MEDIUM", "LOW"
        ]

    def test_triage_routes_high_risk_to_investigate(self):
        """High risk correctly routed to investigation"""
        state = make_high_risk_state()
        routing = self.graph._route_after_triage(
            {**state, "triage_verdict": "INVESTIGATE"}
        )
        assert routing == "investigate"

    def test_triage_routes_low_risk_to_monitor(self):
        """Monitor verdict correctly routed"""
        state = make_low_risk_state()
        routing = self.graph._route_after_triage(
            {**state, "triage_verdict": "MONITOR"}
        )
        assert routing == "monitor"

    def test_triage_routes_false_positive_to_close(self):
        """Close verdict correctly routed"""
        state = make_low_risk_state()
        routing = self.graph._route_after_triage(
            {**state, "triage_verdict": "CLOSE"}
        )
        assert routing == "close"

    def test_compromise_routes_to_response(self):
        """Confirmed compromise routes to response"""
        state = make_high_risk_state()
        routing = self.graph._route_after_investigation(
            {
                **state,
                "compromise_confirmed": True,
                "c2_confirmed": True
            }
        )
        assert routing == "respond"

    def test_no_compromise_routes_to_report(self):
        """No compromise routes directly to report"""
        state = make_low_risk_state()
        routing = self.graph._route_after_investigation(
            {
                **state,
                "compromise_confirmed": False,
                "c2_confirmed": False
            }
        )
        assert routing == "report"