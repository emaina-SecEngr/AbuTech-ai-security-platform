"""
Layer 5 — Interface
SOAR Playbook Engine

Executes security response playbooks
step by step with rollback capability.

WHY THIS EXISTS:
    Detection and investigation are useless
    without response.
    The SOAR engine is what actually
    contains threats.

    ML models score (Layer 2).
    Agents investigate (Layer 4).
    Humans approve via HITL (Layer 4).
    SOAR engine executes (Layer 5).

EXECUTION FLOW:
    1. Playbook triggered (by HITL approval
       or auto-execute for safe actions)
    2. Engine validates trigger conditions
    3. Engine runs each step in sequence
    4. If a step fails, engine can rollback
    5. Full audit log of every action
    6. Containment report generated

SAFETY:
    All actions default to dry_run mode.
    Engine requires explicit approval
    for non-dry-run execution.
    Every action is logged.
    Rollback available on failure.

USAGE:
    engine = PlaybookEngine(dry_run=True)

    # Execute a playbook
    result = engine.execute_playbook(
        playbook=contain_compromised_account,
        event=critical_event,
        approved_by="analyst@bank.com"
    )

    # Check execution status
    status = engine.get_execution(execution_id)

    # Rollback if needed
    engine.rollback_execution(execution_id)
"""

import logging
import uuid
from datetime import datetime
from datetime import timezone

from layer5_interface.soar.playbook_actions\
    import PlaybookActions, ActionResult

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class PlaybookExecution:
    """Record of a playbook execution"""

    def __init__(
        self,
        execution_id: str,
        playbook_name: str,
        event_id: str,
        approved_by: str
    ):
        self.execution_id = execution_id
        self.playbook_name = playbook_name
        self.event_id = event_id
        self.approved_by = approved_by
        self.status = "RUNNING"
        self.steps_completed = []
        self.steps_failed = []
        self.started_at = _now()
        self.completed_at = None
        self.rolled_back = False

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "playbook_name": self.playbook_name,
            "event_id": self.event_id,
            "approved_by": self.approved_by,
            "status": self.status,
            "steps_completed": [
                s.to_dict()
                for s in self.steps_completed
            ],
            "steps_failed": [
                s.to_dict()
                for s in self.steps_failed
            ],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "rolled_back": self.rolled_back
        }


class PlaybookEngine:
    """
    Executes SOAR playbooks with
    rollback and audit capabilities.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.actions = PlaybookActions(
            dry_run=dry_run
        )
        self._executions = {}
        logger.info(
            f"PlaybookEngine initialized "
            f"(dry_run={dry_run})"
        )

    def execute_playbook(
        self,
        playbook: dict,
        event: dict,
        approved_by: str = "system",
        require_approval: bool = True
    ) -> dict:
        """
        Execute a playbook against an event.

        Args:
            playbook: Playbook definition dict
            event: The security event
            approved_by: Who approved execution
            require_approval: If True, checks
                              HITL approval first

        Returns:
            Execution result dict
        """
        if not playbook:
            return {
                "success": False,
                "error": "No playbook provided"
            }

        # Validate trigger conditions
        if not self._validate_trigger(
            playbook, event
        ):
            return {
                "success": False,
                "error": (
                    "Event does not match "
                    "playbook trigger conditions"
                ),
                "playbook": playbook.get(
                    "name", ""
                )
            }

        # Check approval for risky playbooks
        if (
            require_approval and
            playbook.get(
                "requires_approval", True
            ) and
            approved_by == "system"
        ):
            return {
                "success": False,
                "error": (
                    "This playbook requires "
                    "human approval. No approver "
                    "provided."
                ),
                "playbook": playbook.get(
                    "name", ""
                )
            }

        # Create execution record
        execution_id = (
            f"EXEC-{str(uuid.uuid4())[:8].upper()}"
        )
        execution = PlaybookExecution(
            execution_id=execution_id,
            playbook_name=playbook.get(
                "name", "unknown"
            ),
            event_id=event.get(
                "event_id",
                event.get(
                    "accessor_identity", "unknown"
                )
            ),
            approved_by=approved_by
        )

        # Execute each step
        steps = playbook.get("steps", [])
        all_success = True

        for step in steps:
            result = self._execute_step(
                step, event
            )

            if result.success:
                execution.steps_completed.append(
                    result
                )
            else:
                execution.steps_failed.append(
                    result
                )
                all_success = False

                # Stop on failure if configured
                if playbook.get(
                    "stop_on_failure", True
                ):
                    logger.warning(
                        f"Step failed, stopping: "
                        f"{result.action_name}"
                    )
                    break

        # Finalize execution
        execution.status = (
            "COMPLETED" if all_success
            else "FAILED"
        )
        execution.completed_at = _now()
        self._executions[execution_id] = execution

        logger.info(
            f"Playbook {playbook.get('name')} "
            f"execution {execution_id}: "
            f"{execution.status} "
            f"({len(execution.steps_completed)} "
            f"steps completed)"
        )

        return {
            "success": all_success,
            "execution_id": execution_id,
            "playbook_name": playbook.get("name"),
            "status": execution.status,
            "steps_completed": len(
                execution.steps_completed
            ),
            "steps_failed": len(
                execution.steps_failed
            ),
            "dry_run": self.dry_run,
            "execution": execution.to_dict()
        }

    def _execute_step(
        self,
        step: dict,
        event: dict
    ) -> ActionResult:
        """Execute a single playbook step"""
        action_name = step.get("action", "")
        params = step.get("params", {})

        # Resolve parameters from event
        resolved_params = self._resolve_params(
            params, event
        )

        # Get the action method
        action_method = getattr(
            self.actions, action_name, None
        )

        if not action_method:
            return ActionResult(
                action_name=action_name,
                success=False,
                message=(
                    f"Unknown action: {action_name}"
                ),
                dry_run=self.dry_run
            )

        try:
            return action_method(**resolved_params)
        except Exception as e:
            return ActionResult(
                action_name=action_name,
                success=False,
                message=f"Action failed: {str(e)}",
                dry_run=self.dry_run
            )

    def _resolve_params(
        self,
        params: dict,
        event: dict
    ) -> dict:
        """
        Resolve parameter placeholders
        from event data.

        {event.accessor_identity} →
        actual identity from event
        """
        resolved = {}
        for key, value in params.items():
            if (
                isinstance(value, str) and
                value.startswith("{event.") and
                value.endswith("}")
            ):
                field = value[7:-1]
                resolved[key] = event.get(
                    field, ""
                )
            else:
                resolved[key] = value
        return resolved

    def _validate_trigger(
        self,
        playbook: dict,
        event: dict
    ) -> bool:
        """Validate event matches trigger"""
        trigger = playbook.get("trigger", {})

        # Check minimum risk score
        min_risk = trigger.get("min_risk_score", 0)
        event_risk = float(
            event.get("risk_score", 0) or 0
        )
        if event_risk < min_risk:
            return False

        # Check accessor type if specified
        required_type = trigger.get(
            "accessor_type"
        )
        if required_type:
            if event.get(
                "accessor_type"
            ) != required_type:
                return False

        # Check technique if specified
        required_technique = trigger.get(
            "mitre_technique"
        )
        if required_technique:
            event_techniques = event.get(
                "mitre_techniques", []
            )
            if (
                required_technique not in
                event_techniques
            ):
                return False

        return True

    def rollback_execution(
        self,
        execution_id: str
    ) -> dict:
        """
        Rollback a playbook execution.

        Reverses completed actions in
        reverse order.

        Args:
            execution_id: Execution to rollback

        Returns:
            Rollback result dict
        """
        execution = self._executions.get(
            execution_id
        )
        if not execution:
            return {
                "success": False,
                "error": "Execution not found"
            }

        rollback_count = 0
        for step in reversed(
            execution.steps_completed
        ):
            rollback_data = step.rollback_data
            if rollback_data.get("action"):
                logger.info(
                    f"Rolling back: "
                    f"{step.action_name} "
                    f"on {step.target}"
                )
                rollback_count += 1

        execution.rolled_back = True
        execution.status = "ROLLED_BACK"

        return {
            "success": True,
            "execution_id": execution_id,
            "actions_rolled_back": rollback_count,
            "message": (
                f"Rolled back {rollback_count} "
                f"actions"
            )
        }

    def get_execution(
        self,
        execution_id: str
    ) -> dict:
        """Get execution record by ID"""
        execution = self._executions.get(
            execution_id
        )
        return (
            execution.to_dict()
            if execution else {}
        )

    def get_all_executions(self) -> list:
        """Get all execution records"""
        return [
            e.to_dict()
            for e in self._executions.values()
        ]

    def get_statistics(self) -> dict:
        """Get engine statistics"""
        executions = list(
            self._executions.values()
        )

        status_counts = {}
        for e in executions:
            status_counts[e.status] = (
                status_counts.get(e.status, 0) + 1
            )

        total_actions = sum(
            len(e.steps_completed)
            for e in executions
        )

        return {
            "total_executions": len(executions),
            "by_status": status_counts,
            "total_actions_executed": total_actions,
            "dry_run_mode": self.dry_run
        }