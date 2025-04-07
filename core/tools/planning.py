# tool/planning.py
from typing import Dict, List, Literal, Optional

from core.tools.errors import ToolError
from core.tools.base import BaseTool, ToolResult


_PLANNING_TOOL_DESCRIPTION = """
一个规划工具，允许代理创建和管理解决复杂任务的计划。
该工具提供创建计划、更新计划步骤和跟踪进度的功能。
"""


class PlanningTool(BaseTool):
    """
    一个规划工具，允许代理创建和管理解决复杂任务的计划。
    该工具提供创建计划、更新计划步骤和跟踪进度的功能。
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可用命令：create, update, list, get, set_active, mark_step, delete.",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "计划的唯一标识符。在create、update、set_active和delete命令中是必需的。在get和mark_step命令中是可选的（如果未指定，则使用当前活动计划）。",
                "type": "string",
            },
            "title": {
                "description": "计划的标题。在create命令中是必需的，在update命令中是可选的。",
                "type": "string",
            },
            "steps": {
                "description": "计划的步骤列表。在create命令中是必需的，在update命令中是可选的。",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "要更新的步骤的索引（从0开始）。在mark_step命令中是必需的。",
                "type": "integer",
            },
            "step_status": {
                "description": "要设置的步骤的状态。在mark_step命令中使用。",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "步骤的附加注释。在mark_step命令中是可选的。",
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    plans: dict = {}  # Dictionary to store plans by plan_id
    _current_plan_id: Optional[str] = None  # Track the current active plan

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        使用给定的命令和参数执行规划工具。

        Parameters:
        - command: 要执行的操作
        - plan_id: 计划的唯一标识符
        - title: 计划的标题（在create命令中使用）
        - steps: 计划的步骤列表（在create命令中使用）
        - step_index: 要更新的步骤的索引（在mark_step命令中使用）
        - step_status: 要设置的步骤的状态（在mark_step命令中使用）
        - step_notes: 步骤的附加注释（在mark_step命令中使用）
        """

        if command == "create":
            return self._create_plan(plan_id, title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise ToolError(
                f"未识别的命令: {command}. 允许的命令是: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """创建一个具有给定ID、标题和步骤的新计划。"""
        if not plan_id:
            raise ToolError("参数 `plan_id` 在命令: create 中是必需的")

        if plan_id in self.plans:
            raise ToolError(
                f"一个具有ID '{plan_id}' 的计划已经存在。使用 'update' 来修改现有的计划。"
            )

        if not title:
            raise ToolError("参数 `title` 在命令: create 中是必需的")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError(
                "参数 `steps` 在命令: create 中必须是非空字符串列表"
            )

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        return ToolResult(
            output=f"计划创建成功，ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """更新一个具有新标题或步骤的现有计划。"""
        if not plan_id:
            raise ToolError("参数 `plan_id` 在命令: update 中是必需的")

        if plan_id not in self.plans:
            raise ToolError(f"未找到具有ID: {plan_id} 的计划")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError(
                    "参数 `steps` 在命令: update 中必须是一个字符串列表"
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return ToolResult(
            output=f"计划更新成功: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """列出所有可用的计划。"""
        if not self.plans:
            return ToolResult(
                output="没有可用的计划。使用 'create' 命令创建一个计划。"
            )

        output = "可用的计划:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} 步骤已完成"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """获取特定计划的详细信息。"""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "没有活动的计划。请指定一个plan_id或设置一个活动的计划。"
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"未找到具有ID: {plan_id} 的计划")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """设置一个计划为活动的计划。"""
        if not plan_id:
            raise ToolError("参数 `plan_id` 在命令: set_active 中是必需的")

        if plan_id not in self.plans:
            raise ToolError(f"未找到具有ID: {plan_id} 的计划")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"计划 '{plan_id}' 现在是活动的计划。\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """标记一个具有特定状态和可选注释的步骤。"""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "没有活动的计划。请指定一个plan_id或设置一个活动的计划。"
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"未找到具有ID: {plan_id} 的计划")

        if step_index is None:
            raise ToolError("参数 `step_index` 在命令: mark_step 中是必需的")

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"无效的step_index: {step_index}. 有效的索引范围从0到{len(plan['steps'])-1}."
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"无效的step_status: {step_status}. 有效的状态是: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return ToolResult(
            output=f"步骤 {step_index} 在计划 '{plan_id}' 中更新。\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """Delete a plan."""
        if not plan_id:
            raise ToolError("参数 `plan_id` 在命令: delete 中是必需的")

        if plan_id not in self.plans:
            raise ToolError(f"未找到具有ID: {plan_id} 的计划")

        del self.plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"计划 '{plan_id}' 已被删除。")

    def _format_plan(self, plan: Dict) -> str:
        """格式化一个计划以供显示。"""
        output = f"计划: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"进度: {completed}/{total_steps} 步骤已完成 "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"状态: {completed} 已完成, {in_progress} 进行中, {blocked} 阻塞, {not_started} 未开始\n\n"
        output += "步骤:\n"

        # Add each step with its status and notes
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   备注: {notes}\n"

        return output