"""Custom Gradio UI for the FlowOS Space."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr


def _pretty_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _safe_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    return json.loads(text)


def build_developer_control_room_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str = "FlowOS",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    del action_fields, is_chat_env

    display_title = title

    css = """
    :root {
      --bg: #09111f;
      --bg-2: #0f1728;
      --surface: rgba(14, 23, 39, 0.62);
      --surface-strong: rgba(17, 24, 39, 0.88);
      --line: rgba(148, 163, 184, 0.18);
      --text: #eef4ff;
      --muted: #9db0c7;
      --cyan: #6ee7f9;
      --blue: #60a5fa;
      --coral: #fb7185;
      --glow: rgba(110, 231, 249, 0.18);
    }
    body, .gradio-container {
      background:
        radial-gradient(circle at 15% 20%, rgba(96, 165, 250, 0.18), transparent 28%),
        radial-gradient(circle at 85% 12%, rgba(251, 113, 133, 0.16), transparent 24%),
        linear-gradient(180deg, var(--bg), var(--bg-2)) !important;
      color: var(--text);
    }
    .gradio-container { max-width: 1400px !important; padding-top: 18px !important; }
    .dcr-hero {
      padding: 28px 30px 18px 30px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(15, 23, 40, 0.92), rgba(12, 18, 31, 0.72));
      box-shadow: 0 20px 80px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(14px);
      margin-bottom: 18px;
    }
    .dcr-kicker {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      color: var(--cyan);
      font-size: 12px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    .dcr-title {
      font-size: 42px;
      line-height: 1.02;
      font-weight: 760;
      letter-spacing: -0.03em;
      margin: 0;
    }
    .dcr-subtitle {
      max-width: 760px;
      color: var(--muted);
      margin-top: 12px;
      font-size: 16px;
      line-height: 1.65;
    }
    .dcr-panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px 18px 10px 18px;
      backdrop-filter: blur(10px);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .dcr-panel-title {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--cyan);
      margin-bottom: 8px;
    }
    .dcr-side-note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      margin-top: 4px;
      margin-bottom: 10px;
    }
    .dcr-status {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(96, 165, 250, 0.09);
      border: 1px solid rgba(96, 165, 250, 0.16);
    }
    .dcr-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }
    .dcr-chip {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      font-size: 12px;
    }
    .dcr-chip strong { color: var(--cyan); font-weight: 600; }
    .dcr-btn-primary button {
      background: linear-gradient(135deg, var(--cyan), var(--blue));
      border: none !important;
      color: #06111f !important;
      font-weight: 700;
      box-shadow: 0 10px 30px var(--glow);
    }
    .dcr-btn-secondary button {
      background: rgba(255, 255, 255, 0.03) !important;
      border: 1px solid var(--line) !important;
      color: var(--text) !important;
    }
    .dcr-btn-secondary button:hover,
    .dcr-btn-primary button:hover {
      filter: brightness(1.05);
    }
    .dcr-json, .dcr-json textarea, .dcr-json pre {
      font-size: 12px !important;
    }
    .block, .gr-group, .gr-box, .gr-panel {
      border-radius: 20px !important;
    }
    """

    async def reset_env(task_id: str, scenario_index: int):
        try:
            data = await web_manager.reset_environment(
                {"task_id": task_id, "scenario_index": scenario_index}
            )
            return (
                _pretty_json(data.get("observation", {})),
                _pretty_json(data),
                "Reset complete.",
            )
        except Exception as exc:
            return ("", "", f"Reset error: {exc}")

    async def step_env(action_type: str, params_json: str):
        try:
            parameters = _safe_json(params_json)
        except Exception as exc:
            return ("", "", f"Invalid JSON: {exc}")

        payload = {"action_type": action_type, "parameters": parameters}
        try:
            data = await web_manager.step_environment(payload)
            return (
                _pretty_json(data.get("observation", {})),
                _pretty_json(data),
                "Step complete.",
            )
        except Exception as exc:
            return ("", "", f"Step error: {exc}")

    def get_state_sync():
        try:
            return _pretty_json(web_manager.get_state())
        except Exception as exc:
            return f"State error: {exc}"

    with gr.Blocks(title=display_title) as demo:
        gr.HTML(f"<style>{css}</style>")
        gr.HTML(
            f"""
            <section class="dcr-hero">
              <div class="dcr-kicker">FlowOS / Interactive Workspace / End-to-End Ops</div>
              <div class="dcr-title">{display_title}</div>
              <div class="dcr-subtitle">
                Explore repair, review, and shipping in one live operating environment.
                FlowOS is built to evaluate how AI agents move real work from broken to shipped.
              </div>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.HTML("<div class='dcr-chip-row'><span class='dcr-chip'><strong>Port</strong> 7860</span><span class='dcr-chip'><strong>Mode</strong> Single-page UI</span></div>")
                with gr.Group(elem_classes="dcr-panel"):
                    gr.HTML("<div class='dcr-panel-title'>Scenario Control</div>")
                    gr.HTML("<div class='dcr-side-note'>Pick a FlowOS task, reset the episode, then drive the environment with direct JSON actions.</div>")
                    task_id = gr.Dropdown(
                        choices=[
                            "repair_data_transform",
                            "repair_pipeline_execution",
                            "review_ai_patch_safety",
                            "review_ai_patch_correctness",
                            "synthesize_reporting_asset",
                            "synthesize_data_product",
                        ],
                        value="repair_data_transform",
                        label="Task",
                    )
                    scenario_index = gr.Number(value=0, label="Scenario Index", precision=0)
                    reset_btn = gr.Button("Reset Episode", elem_classes="dcr-btn-secondary")

                with gr.Group(elem_classes="dcr-panel"):
                    gr.HTML("<div class='dcr-panel-title'>Action Composer</div>")
                    gr.HTML("<div class='dcr-side-note'>Use exact action names and JSON params. This is intentionally close to the raw FlowOS benchmark surface.</div>")
                    action_type = gr.Dropdown(
                        choices=[
                            "search_workspace",
                            "read_file",
                            "inspect_schema",
                            "inspect_lineage",
                            "inspect_llm_draft",
                            "edit_file",
                            "run_validator",
                            "submit_repair",
                            "submit_review",
                            "submit_workspace",
                        ],
                        value="read_file",
                        label="Action Type",
                    )
                    params_json = gr.Code(
                        value='{"path":"transforms/orders_daily.sql"}',
                        label="Action Parameters (JSON)",
                        language="json",
                        elem_classes="dcr-json",
                    )
                    with gr.Row():
                        step_btn = gr.Button("Send Action", elem_classes="dcr-btn-primary")
                        state_btn = gr.Button("Get State", elem_classes="dcr-btn-secondary")

                with gr.Group(elem_classes="dcr-status"):
                    status = gr.Textbox(label="Status", interactive=False, container=False)

            with gr.Column(scale=4):
                with gr.Group(elem_classes="dcr-panel"):
                    gr.HTML("<div class='dcr-panel-title'>Observation</div>")
                    obs_display = gr.Code(label="Observation", language="json", value="")
                with gr.Group(elem_classes="dcr-panel"):
                    gr.HTML("<div class='dcr-panel-title'>Raw JSON</div>")
                    raw_json = gr.Code(label="Raw Response", language="json", value="")
                if quick_start_md:
                    with gr.Group(elem_classes="dcr-panel"):
                        gr.HTML("<div class='dcr-panel-title'>Quick Start</div>")
                        gr.Markdown(quick_start_md)

        reset_btn.click(
            fn=reset_env,
            inputs=[task_id, scenario_index],
            outputs=[obs_display, raw_json, status],
        )
        step_btn.click(
            fn=step_env,
            inputs=[action_type, params_json],
            outputs=[obs_display, raw_json, status],
        )
        state_btn.click(
            fn=get_state_sync,
            outputs=[raw_json],
        )

    return demo
