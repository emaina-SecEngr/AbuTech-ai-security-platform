"""
Layer 4 — LLM Reasoning
Specialist Security Agents

Five specialist agents that collaborate
to investigate security alerts:

    TriageAgent       → Is this worth investigating?
    IntelAgent        → Who is behind this attack?
    InvestigationAgent → What happened exactly?
    ResponseAgent     → What should we do?
    ReportAgent       → Document the findings
"""

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import Optional

from layer4_reasoning.agents.agent_state import (
    InvestigationState
)

# Agent tools for real data access
try:
    from layer4_reasoning.tools.tool_registry import (
        check_ip_reputation,
        query_knowledge_graph,
        get_ensemble_scores,
        search_past_incidents,
        store_incident,
        get_incident_stats,
        get_user_permissions
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    
    # Agent memory for persistent storage
try:
    from layer4_reasoning.memory.memory_store import (
        AgentMemoryStore
    )
    MEMORY_AVAILABLE = True
    _memory_store = AgentMemoryStore()
except Exception:
    MEMORY_AVAILABLE = False
    _memory_store = None

logger = logging.getLogger(__name__)


# ============================================================
# THREAT ACTOR KNOWLEDGE BASE
# ============================================================

THREAT_ACTOR_PROFILES = {
    "TA542": {
        "name": "Mummy Spider (Emotet)",
        "motivation": "financial",
        "ttps": ["T1566.001", "T1059.001", "T1127", "T1568.002", "T1071.001"],
        "indicators": ["msbuild spawning powershell", "encoded powershell", "duckdns domains", "tor exit node c2"],
        "description": "Financially motivated group operating Emotet botnet."
    },
    "APT29": {
        "name": "Cozy Bear",
        "motivation": "espionage",
        "ttps": ["T1566.002", "T1059.001", "T1078", "T1021.001"],
        "indicators": ["spearphishing links", "valid account abuse", "remote desktop"],
        "description": "Russian SVR-linked espionage group targeting government."
    }
}


# ============================================================
# BASE AGENT
# ============================================================

class BaseSecurityAgent:

    def __init__(self, agent_name: str, anthropic_api_key: str = None):
        self.agent_name = agent_name
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.use_llm = bool(self.api_key)
        logger.info(f"{agent_name} initialized ({'LLM' if self.use_llm else 'Rule'} mode)")

    def _log(self, state: InvestigationState, message: str) -> list:
        entry = {
            "agent": self.agent_name,
            "timestamp": self._now(),
            "message": message
        }
        logger.info(f"[{self.agent_name}] {message}")
        return state.get("agent_log", []) + [entry]

    def _call_claude(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        if not self.api_key:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            logger.warning(f"Claude API call failed: {e}")
            return None

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ============================================================
# TRIAGE AGENT
# ============================================================

class TriageAgent(BaseSecurityAgent):

    def __init__(self, anthropic_api_key: str = None):
        super().__init__("TriageAgent", anthropic_api_key)

    def run(self, state: InvestigationState) -> InvestigationState:
        risk = state["overall_risk_score"]
        triage_verdict, priority, confidence = self._evaluate_alert(state)
        reasoning = self._build_reasoning(state, triage_verdict, priority)

        if self.use_llm:
            llm_reasoning = self._llm_triage(state)
            if llm_reasoning:
                reasoning = llm_reasoning

        log_entries = self._log(state, f"Triage complete: {triage_verdict} ({priority}) risk={risk:.2f}")

        return {
            **state,
            "triage_verdict": triage_verdict,
            "triage_priority": priority,
            "triage_confidence": confidence,
            "triage_reasoning": reasoning,
            "agent_log": log_entries
        }

    def _evaluate_alert(self, state: InvestigationState) -> tuple:
        risk = state["overall_risk_score"]
        verdict = state["overall_verdict"]

        if verdict == "MALWARE" and (state.get("malware_risk") or 0) >= 0.8:
            return "INVESTIGATE", "CRITICAL", 0.95
        if verdict == "DGA_DOMAIN" and (state.get("dga_risk") or 0) >= 0.8:
            return "INVESTIGATE", "HIGH", 0.90
        if risk >= 0.8:
            return "INVESTIGATE", "CRITICAL", 0.90
        elif risk >= 0.6:
            return "INVESTIGATE", "HIGH", 0.85
        elif risk >= 0.4:
            return "MONITOR", "MEDIUM", 0.75
        elif risk >= 0.2:
            return "MONITOR", "LOW", 0.70
        else:
            return "CLOSE", "LOW", 0.85

    def _build_reasoning(self, state: InvestigationState, verdict: str, priority: str) -> str:
        risk = state["overall_risk_score"]
        host = state["event_host"]
        ml_verdict = state["overall_verdict"]

        reasoning = f"Alert on {host} scored {risk:.2f} overall. ML verdict: {ml_verdict}. "

        malware_risk = state.get("malware_risk")
        if malware_risk and isinstance(malware_risk, (int, float)):
            reasoning += f"Malware score: {malware_risk:.2f}. "

        dga_risk = state.get("dga_risk")
        if dga_risk and isinstance(dga_risk, (int, float)):
            reasoning += f"DGA score: {dga_risk:.2f}. "

        reasoning += f"Graph shows {state['graph_node_count']} connected entities. Decision: {verdict} ({priority})."
        return reasoning

    def _llm_triage(self, state: InvestigationState) -> Optional[str]:
        prompt = f"""You are a SOC triage analyst.
Evaluate this security alert and provide a brief triage assessment.

Alert Details:
- Host: {state['event_host']}
- User: {state['event_user']}
- Overall Risk: {state['overall_risk_score']:.2f}
- ML Verdict: {state['overall_verdict']}
- Malware Risk: {state.get('malware_risk', 'N/A')}
- DGA Risk: {state.get('dga_risk', 'N/A')}
- ATT&CK Techniques: {state.get('attack_techniques', [])}
- Graph Connections: {state['graph_node_count']} nodes

Provide a 2-3 sentence triage assessment explaining
why this alert should be investigated or closed.
Be specific about which indicators are most concerning."""
        return self._call_claude(prompt, max_tokens=300)


# ============================================================
# INTEL AGENT
# ============================================================

class IntelAgent(BaseSecurityAgent):

    def __init__(self, anthropic_api_key: str = None):
        super().__init__("IntelAgent", anthropic_api_key)

    def run(self, state: InvestigationState) -> InvestigationState:
        actor, confidence = self._attribute_threat_actor(state)
        c2_confirmed = self._check_c2_confirmation(state)
        malware_family = self._identify_malware_family(state)
        intel_summary = self._build_intel_summary(state, actor, malware_family, c2_confirmed)

        if self.use_llm:
            llm_summary = self._llm_intel_analysis(state)
            if llm_summary:
                intel_summary = llm_summary

        confirmed_techniques = self._map_attack_techniques(state, actor)
        log_entries = self._log(state, f"Intel: actor={actor} malware={malware_family} c2={c2_confirmed}")

        return {
            **state,
            "threat_actor_identified": actor,
            "threat_actor_confidence": confidence,
            "c2_confirmed": c2_confirmed,
            "malware_family_confirmed": malware_family,
            "intel_summary": intel_summary,
            "confirmed_techniques": confirmed_techniques,
            "agent_log": log_entries
        }

    def _attribute_threat_actor(self, state: InvestigationState) -> tuple:
        indicators_text = " ".join([
            str(state.get("malware_indicators", [])),
            str(state.get("attack_techniques", [])),
            str(state.get("dga_indicators", [])),
            state.get("overall_verdict", "")
        ]).lower()

        best_match = None
        best_score = 0

        for actor_id, profile in THREAT_ACTOR_PROFILES.items():
            score = sum(1 for ind in profile["indicators"] if ind in indicators_text)
            match_ratio = score / len(profile["indicators"])
            if match_ratio > best_score:
                best_score = match_ratio
                best_match = actor_id

        if best_score >= 0.5:
            confidence = min(0.5 + best_score * 0.4, 0.95)
            return best_match, confidence
        return None, 0.0

    def _check_c2_confirmation(self, state: InvestigationState) -> bool:
        dga_risk = state.get("dga_risk") or 0
        has_dga = dga_risk >= 0.7
        has_c2_ip = any(
            e.get("risk", 0) >= 0.7
            for e in state.get("threat_connections", [])
            if e.get("type") == "ip_address"
        )
        has_malware = state.get("overall_verdict") == "MALWARE"
        return has_malware and (has_dga or has_c2_ip)

    def _identify_malware_family(self, state: InvestigationState) -> Optional[str]:
        if state.get("known_malware_family"):
            return state["known_malware_family"]
        indicators = str(state.get("malware_indicators", [])).lower()
        if "emotet" in indicators:
            return "Emotet"
        if "cobalt" in indicators or "beacon" in indicators:
            return "Cobalt Strike"
        if "trickbot" in indicators:
            return "TrickBot"
        techniques = state.get("attack_techniques", [])
        if "T1566.001" in techniques and "T1059.001" in techniques:
            return "Unknown - Emotet-like pattern"
        return None

    def _map_attack_techniques(self, state: InvestigationState, actor: Optional[str]) -> list:
        techniques = list(state.get("attack_techniques", []))
        if actor and actor in THREAT_ACTOR_PROFILES:
            for ttp in THREAT_ACTOR_PROFILES[actor]["ttps"]:
                if ttp not in techniques:
                    techniques.append(ttp)
        return techniques

    def _build_intel_summary(self, state: InvestigationState, actor: Optional[str], malware: Optional[str], c2_confirmed: bool) -> str:
        host = state["event_host"]
        summary = f"Threat intelligence for {host}: "
        if actor and actor in THREAT_ACTOR_PROFILES:
            profile = THREAT_ACTOR_PROFILES[actor]
            summary += f"Activity attributed to {profile['name']} ({actor}). {profile['description']} "
        if malware:
            summary += f"Malware family: {malware}. "
        if c2_confirmed:
            summary += "C2 communication confirmed — active compromise in progress. "
        return summary

    def _llm_intel_analysis(self, state: InvestigationState) -> Optional[str]:
        """Use Claude + real tools for threat intel"""
        # Get real IP reputation
        ip_intel = {}
        source_ip = state.get("event_host", "")
        if TOOLS_AVAILABLE and source_ip:
            try:
                ip_intel = check_ip_reputation(source_ip)
            except Exception:
                pass

        # Get real graph context
        graph_intel = {}
        if TOOLS_AVAILABLE:
            try:
                graph_intel = query_knowledge_graph(
                    state.get("event_user", "")
                )
            except Exception:
                pass

        # Search past incidents
        # Search past incidents using vector memory
        past_incidents = []
        if MEMORY_AVAILABLE and _memory_store:
            try:
                query = (
                    f"{state.get('event_user', '')} "
                    f"{state.get('overall_verdict', '')} "
                    f"{state.get('event_host', '')}"
                )
                past_incidents = _memory_store.search_similar(
                    query, n_results=3
                )
            except Exception:
                pass
        elif TOOLS_AVAILABLE:
            try:
                past_incidents = search_past_incidents(
                    accessor=state.get("event_user", ""),
                    pattern=state.get("overall_verdict", "")
                )
            except Exception:
                pass

        prompt = f"""You are a threat intelligence analyst.
Analyze these indicators and provide attribution.

Event Details:
- Host: {state['event_host']}
- User: {state['event_user']}
- Verdict: {state.get('overall_verdict', 'UNKNOWN')}

IP Intelligence:
- Score: {ip_intel.get('score', 'N/A')}/100
- Is Tor: {ip_intel.get('is_tor', 'Unknown')}
- Categories: {ip_intel.get('categories', [])}
- Summary: {ip_intel.get('summary', 'No data')}

Knowledge Graph:
- Entity risk: {graph_intel.get('risk_score', 'N/A')}
- Connections: {len(graph_intel.get('connections', []))}
- Threat proximity: {graph_intel.get('threat_proximity', 'N/A')}
- Summary: {graph_intel.get('summary', 'No data')}

Past Incidents:
- Similar incidents found: {len(past_incidents)}
- Repeat offender: {len(past_incidents) >= 3}

ATT&CK Techniques: {state.get('attack_techniques', [])}
Known Malware: {state.get('known_malware_family', 'Unknown')}

Provide a 3-4 sentence threat intelligence assessment:
1. Threat actor attribution with reasoning
2. Whether this is a repeat pattern
3. C2 and infrastructure assessment
4. Confidence level"""

        return self._call_claude(prompt, max_tokens=400)


# ============================================================
# INVESTIGATION AGENT
# ============================================================

class InvestigationAgent(BaseSecurityAgent):

    def __init__(self, anthropic_api_key: str = None):
        super().__init__("InvestigationAgent", anthropic_api_key)

    def run(self, state: InvestigationState) -> InvestigationState:
        timeline = self._build_timeline(state)
        compromise = self._confirm_compromise(state)
        access_vector = self._determine_access_vector(state)
        blast_radius = self._calculate_blast_radius(state)
        lateral_movement = self._check_lateral_movement(state)
        exfiltration = self._check_exfiltration(state)
        summary = self._build_investigation_summary(state, timeline, compromise, access_vector, blast_radius)

        if self.use_llm:
            llm_summary = self._llm_investigation_analysis(state)
            if llm_summary:
                summary = llm_summary

        log_entries = self._log(state, f"Investigation: compromise={compromise} timeline={len(timeline)} blast_radius={len(blast_radius)}")

        return {
            **state,
            "attack_timeline": timeline,
            "compromise_confirmed": compromise,
            "initial_access_vector": access_vector,
            "lateral_movement_detected": lateral_movement,
            "data_exfiltration_suspected": exfiltration,
            "blast_radius": blast_radius,
            "investigation_summary": summary,
            "agent_log": log_entries
        }

    def _build_timeline(self, state: InvestigationState) -> list:
        timeline = []
        timestamp = state.get("event_timestamp", "")
        techniques = state.get("confirmed_techniques", []) or state.get("attack_techniques", [])

        if "T1566.001" in techniques:
            timeline.append({"timestamp": timestamp, "event": "Phishing email delivered", "technique": "T1566.001", "significance": "Initial access vector"})
        if state.get("malware_risk") and state["malware_risk"] > 0.5:
            timeline.append({"timestamp": timestamp, "event": f"Malware execution detected on {state['event_host']}", "technique": "T1059.001", "significance": "Execution phase"})
        if (state.get("dga_risk") or 0) > 0.5:
            timeline.append({"timestamp": timestamp, "event": "DGA beaconing observed", "technique": "T1568.002", "significance": "C2 establishment"})
        if state.get("c2_confirmed"):
            timeline.append({"timestamp": timestamp, "event": "C2 communication confirmed", "technique": "T1071.001", "significance": "Active compromise — attacker has control"})
        return timeline

    def _confirm_compromise(self, state: InvestigationState) -> bool:
        return bool(state.get("c2_confirmed") and state.get("malware_risk", 0) > 0.7)

    def _determine_access_vector(self, state: InvestigationState) -> Optional[str]:
        techniques = state.get("confirmed_techniques", []) or state.get("attack_techniques", [])
        indicators = str(state.get("malware_indicators", [])).lower()
        if "T1566.001" in techniques or "winword" in indicators or "excel" in indicators:
            return "Spearphishing attachment — malicious Office document"
        elif "T1566.002" in techniques:
            return "Spearphishing link"
        elif "T1078" in techniques:
            return "Valid account credentials"
        return "Unknown — investigation ongoing"

    def _calculate_blast_radius(self, state: InvestigationState) -> list:
        blast_radius = []
        if state["event_host"]:
            blast_radius.append({"entity": state["event_host"], "type": "host", "status": "COMPROMISED"})
        if state["event_user"]:
            blast_radius.append({"entity": state["event_user"], "type": "user", "status": "CREDENTIALS_AT_RISK"})
        for entity in state.get("high_risk_entities", [])[:3]:
            blast_radius.append({"entity": entity.get("entity", ""), "type": entity.get("type", ""), "status": "AFFECTED"})
        return blast_radius

    def _check_lateral_movement(self, state: InvestigationState) -> bool:
        techniques = state.get("confirmed_techniques", []) or []
        return any(t in techniques for t in ["T1021.001", "T1021.002", "T1550", "T1075"])

    def _check_exfiltration(self, state: InvestigationState) -> bool:
        techniques = state.get("confirmed_techniques", []) or []
        return any(t in techniques for t in ["T1041", "T1048", "T1567"])

    def _build_investigation_summary(self, state, timeline, compromise, access_vector, blast_radius) -> str:
        host = state["event_host"]
        user = state["event_user"]
        summary = f"Investigation of {host} ({user}): "
        summary += "ACTIVE COMPROMISE CONFIRMED. " if compromise else "Suspicious activity detected. "
        if access_vector:
            summary += f"Initial access: {access_vector}. "
        summary += f"Attack timeline: {len(timeline)} stages. Blast radius: {len(blast_radius)} entities."
        return summary

    def _llm_investigation_analysis(self, state: InvestigationState) -> Optional[str]:
        """Use Claude + real tools for investigation"""
        # Get real ensemble scores
        ensemble = {}
        if TOOLS_AVAILABLE:
            try:
                ensemble = get_ensemble_scores(
                    {
                        "risk_score": state.get("overall_risk_score", 0.5),
                        "accessor_identity": state.get("event_user", ""),
                        "data_store_name": state.get("event_host", ""),
                        "event_time": state.get("event_timestamp", "")
                    },
                    pii_sensitivity="PCI"
                )
            except Exception:
                pass

        # Get real permissions
        permissions = {}
        if TOOLS_AVAILABLE:
            try:
                permissions = get_user_permissions(
                    state.get("event_user", ""),
                    accessed_resource=state.get("event_host", "")
                )
            except Exception:
                pass

        # Get graph context
        graph_data = {}
        if TOOLS_AVAILABLE:
            try:
                graph_data = query_knowledge_graph(
                    state.get("event_host", "")
                )
            except Exception:
                pass

        prompt = f"""You are a forensic security investigator.
Analyze this incident and provide investigation summary.

Incident:
- Host: {state['event_host']}
- User: {state['event_user']}
- Risk Score: {state['overall_risk_score']:.2f}
- Verdict: {state.get('overall_verdict', 'UNKNOWN')}

ML Ensemble Analysis:
- Final Score: {ensemble.get('final_score', 'N/A')}
- Verdict: {ensemble.get('verdict', 'N/A')}
- Explanation: {ensemble.get('explanation', 'N/A')}

IAM Permissions Analysis:
- User Type: {permissions.get('type', 'N/A')}
- Risk Level: {permissions.get('risk_level', 'N/A')}
- Violations: {permissions.get('violations', [])}
- Summary: {permissions.get('summary', 'N/A')}

Knowledge Graph:
- Host risk: {graph_data.get('risk_score', 'N/A')}
- Connections: {len(graph_data.get('connections', []))}

Compromise: {state.get('compromise_confirmed', False)}
C2 Active: {state.get('c2_confirmed', False)}
Techniques: {state.get('confirmed_techniques', [])}

Provide a 4-5 sentence investigation summary:
1. What happened (specific details)
2. How attacker gained access
3. What data is at risk
4. Policy violations identified
5. Confidence in assessment"""

        return self._call_claude(prompt, max_tokens=500)


# ============================================================
# RESPONSE AGENT
# ============================================================

class ResponseAgent(BaseSecurityAgent):

    def __init__(self, anthropic_api_key: str = None):
        super().__init__("ResponseAgent", anthropic_api_key)

    def run(self, state: InvestigationState) -> InvestigationState:
        actions = self._generate_response_actions(state)
        priority = self._determine_containment_priority(state)
        isolation = self._recommend_isolation(state)
        cred_reset = self._recommend_credential_reset(state)
        summary = self._build_response_summary(state, actions, priority)

        if self.use_llm:
            llm_summary = self._llm_response_plan(state)
            if llm_summary:
                summary = llm_summary

        log_entries = self._log(state, f"Response: {len(actions)} actions priority={priority} isolation={isolation}")

        return {
            **state,
            "response_actions": actions,
            "containment_priority": priority,
            "isolation_recommended": isolation,
            "credential_reset_recommended": cred_reset,
            "response_summary": summary,
            "agent_log": log_entries
        }

    def _generate_response_actions(self, state: InvestigationState) -> list:
        actions = []
        if state.get("compromise_confirmed"):
            actions.append({"priority": 1, "action": "ISOLATE_HOST", "target": state["event_host"], "reason": "Active compromise confirmed", "automated": False})
        if state.get("c2_confirmed"):
            actions.append({"priority": 2, "action": "BLOCK_C2_INFRASTRUCTURE", "target": "Network firewall", "reason": "Block C2 IPs and domains", "automated": True})
        if state.get("credential_reset_recommended"):
            actions.append({"priority": 3, "action": "RESET_CREDENTIALS", "target": state["event_user"], "reason": "Credentials may be compromised", "automated": False})
        actions.append({"priority": 4, "action": "PRESERVE_FORENSIC_EVIDENCE", "target": state["event_host"], "reason": "Capture memory and disk image", "automated": False})
        actions.append({"priority": 5, "action": "NOTIFY_SECURITY_TEAM", "target": "SOC Manager", "reason": f"Priority: {state.get('triage_priority', 'HIGH')}", "automated": True})
        return actions

    def _determine_containment_priority(self, state: InvestigationState) -> str:
        if state.get("compromise_confirmed"):
            return "IMMEDIATE"
        elif state.get("c2_confirmed"):
            return "URGENT"
        elif state.get("overall_risk_score", 0) >= 0.7:
            return "HIGH"
        return "NORMAL"

    def _recommend_isolation(self, state: InvestigationState) -> bool:
        return bool(state.get("compromise_confirmed") or (state.get("c2_confirmed") and state.get("overall_risk_score", 0) >= 0.8))

    def _recommend_credential_reset(self, state: InvestigationState) -> bool:
        return bool(state.get("compromise_confirmed") or state.get("lateral_movement_detected"))

    def _build_response_summary(self, state, actions, priority) -> str:
        host = state["event_host"]
        user = state["event_user"]
        summary = f"Response plan for {host} ({user}): Containment priority: {priority}. {len(actions)} actions required. "
        top_actions = [a["action"] for a in sorted(actions, key=lambda x: x["priority"])[:3]]
        summary += f"Top actions: {', '.join(top_actions)}."
        return summary

    def _llm_response_plan(self, state: InvestigationState) -> Optional[str]:
        prompt = f"""You are an incident response expert.
Create a prioritized response plan for this incident.

Incident Summary:
- Host: {state['event_host']}
- User: {state['event_user']}
- Compromise Confirmed: {state.get('compromise_confirmed')}
- C2 Active: {state.get('c2_confirmed')}
- Threat Actor: {state.get('threat_actor_identified')}
- Blast Radius: {state.get('blast_radius', [])}

Provide a 4-5 sentence response plan:
1. Immediate containment actions
2. Evidence preservation steps
3. Credential and access remediation
4. Communication and notification requirements
5. Recovery timeline estimate"""
        return self._call_claude(prompt, max_tokens=400)


# ============================================================
# REPORT AGENT
# ============================================================

class ReportAgent(BaseSecurityAgent):

    def __init__(self, anthropic_api_key: str = None):
        super().__init__("ReportAgent", anthropic_api_key)

    def run(self, state: InvestigationState) -> InvestigationState:
        severity = self._calculate_severity(state)
        executive_summary = self._generate_executive_summary(state)
        full_report = self._generate_full_report(state, severity)

        if self.use_llm:
            llm_report = self._llm_generate_report(state)
            if llm_report:
                full_report = llm_report

        log_entries = self._log(state, f"Report generated: severity={severity}")

        # Store completed investigation for future reference
        if TOOLS_AVAILABLE:
            try:
                store_incident(state)
            except Exception as e:
                logger.debug(f"Could not store incident: {e}")
        # Store in vector memory for semantic search
        if MEMORY_AVAILABLE and _memory_store:
            try:
                _memory_store.store_investigation(state)
            except Exception as e:
                logger.debug(f"Memory store failed: {e}")        

        return {
            **state,
            "severity_rating": severity,
            "executive_summary": executive_summary,
            "final_report": full_report,
            "agent_log": log_entries
        }

    def _calculate_severity(self, state: InvestigationState) -> str:
        if state.get("compromise_confirmed"):
            return "CRITICAL"
        elif state.get("c2_confirmed"):
            return "HIGH"
        elif state.get("overall_risk_score", 0) >= 0.7:
            return "HIGH"
        elif state.get("overall_risk_score", 0) >= 0.4:
            return "MEDIUM"
        return "LOW"

    def _generate_executive_summary(self, state: InvestigationState) -> str:
        host = state["event_host"]
        severity = self._calculate_severity(state)
        actor = state.get("threat_actor_identified", "Unknown actor")
        malware = state.get("malware_family_confirmed", "unknown malware")

        if severity in ["CRITICAL", "HIGH"]:
            return (
                f"{severity} severity incident on {host}. "
                f"Activity attributed to {actor} using {malware}. "
                f"{'Active compromise confirmed. ' if state.get('compromise_confirmed') else ''}"
                f"Immediate response required."
            )
        return f"Monitoring recommended for {host}."

    def _generate_full_report(self, state: InvestigationState, severity: str) -> str:
        report = f"""
{'='*60}
ABUTECH AI SECURITY PLATFORM
INCIDENT INVESTIGATION REPORT
{'='*60}

SEVERITY: {severity}
HOST: {state['event_host']}
USER: {state['event_user']}
TIMESTAMP: {state['event_timestamp']}

EXECUTIVE SUMMARY
{'-'*40}
{state.get('executive_summary', 'N/A')}

TRIAGE ASSESSMENT
{'-'*40}
Verdict: {state.get('triage_verdict', 'N/A')}
Priority: {state.get('triage_priority', 'N/A')}
Reasoning: {state.get('triage_reasoning', 'N/A')}

THREAT INTELLIGENCE
{'-'*40}
Threat Actor: {state.get('threat_actor_identified', 'Unknown')}
Malware Family: {state.get('malware_family_confirmed', 'Unknown')}
C2 Confirmed: {state.get('c2_confirmed', False)}
Intel Summary: {state.get('intel_summary', 'N/A')}

INVESTIGATION FINDINGS
{'-'*40}
Compromise Confirmed: {state.get('compromise_confirmed', False)}
Initial Access: {state.get('initial_access_vector', 'Unknown')}
Lateral Movement: {state.get('lateral_movement_detected', False)}
Exfiltration: {state.get('data_exfiltration_suspected', False)}

Attack Timeline:"""

        for event in state.get("attack_timeline", []):
            report += f"\n  [{event.get('technique', '')}] {event.get('event', '')} — {event.get('significance', '')}"

        report += "\n\nBlast Radius:"
        for entity in state.get("blast_radius", []):
            report += f"\n  {entity.get('entity', '')} ({entity.get('type', '')}) — {entity.get('status', '')}"

        report += f"""

RESPONSE PLAN
{'-'*40}
Containment Priority: {state.get('containment_priority', 'N/A')}
Isolation Recommended: {state.get('isolation_recommended', False)}
Credential Reset: {state.get('credential_reset_recommended', False)}

Response Actions:"""

        for action in state.get("response_actions", []):
            report += f"\n  [{action.get('priority', 0)}] {action.get('action', '')} — {action.get('target', '')}"

        report += f"""

MITRE ATT&CK COVERAGE
{'-'*40}
Techniques Identified: {', '.join(state.get('confirmed_techniques', []))}

{'='*60}
Report generated by AbuTech AI Security Platform
{'='*60}
"""
        return report

    def _llm_generate_report(self, state: InvestigationState) -> Optional[str]:
        prompt = f"""You are a senior security analyst.
Generate a professional incident report based on these findings.

Findings:
- Host: {state['event_host']}
- Severity: {self._calculate_severity(state)}
- Threat Actor: {state.get('threat_actor_identified')}
- Malware: {state.get('malware_family_confirmed')}
- Compromise: {state.get('compromise_confirmed')}
- C2 Active: {state.get('c2_confirmed')}
- Timeline: {state.get('attack_timeline', [])}
- Response: {state.get('response_actions', [])}
- Techniques: {state.get('confirmed_techniques', [])}

Write a professional 6-8 sentence incident report covering:
what happened, how it happened, impact assessment,
and recommended actions. Use clear language."""
        return self._call_claude(prompt, max_tokens=600)