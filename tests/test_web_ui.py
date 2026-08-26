import base64
import re
from pathlib import Path

import kairocli.web_app as web_app_module


def test_web_ui_contains_mode_plan_and_ime_controls() -> None:
    html = (Path(web_app_module.__file__).parent / "web_static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'data-mode="agent"' in html
    assert 'data-mode="agent" title="ReAct 模式：直接执行">ReAct</button>' in html
    assert 'data-mode="plan"' in html
    assert 'data-mode="team"' in html
    assert "<kbd>Enter</kbd> 发送" not in html
    assert "<kbd>Shift</kbd>+<kbd>Enter</kbd> 换行" not in html
    assert 'id="plan-review-overlay"' in html
    assert 'id="model-badge"' in html
    assert 'id="project-list"' in html
    assert 'id="add-workspace-btn"' in html
    assert 'id="workspace-modal"' in html
    sidebar = html.split("<!-- Sidebar -->", 1)[1].split("<!-- Main chat area -->", 1)[0]
    assert 'id="settings-hub-btn"' in sidebar
    assert 'id="admin-btn"' not in sidebar
    assert 'id="config-btn"' not in sidebar
    assert 'id="channel-btn"' not in sidebar
    assert 'id="chpwd-btn"' not in sidebar
    assert 'id="logout-btn"' not in sidebar
    assert 'id="settings-modal"' in html
    assert 'class="settings-list"' in html
    assert '<span class="settings-item-title">用户与额度</span>' in html
    assert '<span class="settings-item-title">模型与服务</span>' in html
    assert '<span class="settings-item-title">微信渠道</span>' in html
    assert "[adminBtn, configBtn, channelBtn, chpwdBtn].forEach" not in html
    assert 'class="admin-row-actions"' in html
    assert '<div class="sidebar-section-label">工作区</div>' in html
    assert "'/v1/workspaces'" in html
    assert "JSON.stringify({ workspace })" in html
    assert "const workspaces = state.workspaces" in html
    assert "暂无可用工作区" in html
    assert "channelWorkspace.disabled = workspaces.length === 0" in html
    assert 'id="channel-workspace-browse"' not in html
    assert "startWorkspaceDraft(state.currentWorkspace)" in html
    assert "createWorkspaceThread(state.currentWorkspace, text)" in html
    assert "eventType === 'thread.title.updated'" in html
    assert "eventType === 'thread.title.failed'" in html
    assert "startEventStream(state.activeThreadId, turn.id)" in html
    assert "follow=true" in html
    assert "res.body.getReader()" in html
    assert "requestAnimationFrame(() => typewriterStep(turnId))" in html
    assert "typewriterComplete(turnId)" in html
    assert "poll(threadId" not in html
    assert "toggleWorkspace(workspace.path)" in html
    assert "state.openWorkspaces[path] = false" in html
    assert "JSON.stringify({ path: state.workspaceBrowserPath })" in html
    assert "threads.slice(0, 4)" in html
    assert "移除项目并清除会话" in html
    assert "在此项目新建会话" in html
    assert "主机文件夹不会被删除" in html
    assert "method: 'DELETE'" in html
    assert "display: flex; flex-direction: column; gap: 3px;" in html
    assert "display: flex; flex-direction: column; gap: 2px;" in html
    assert "e.isComposing" in html
    match = re.search(
        r'<link rel="icon" type="image/svg\+xml" '
        r'href="data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)">',
        html,
    )
    assert match is not None
    favicon = Path(web_app_module.__file__).parent / "web_static" / "favicon.svg"
    assert base64.b64decode(match.group(1)) == favicon.read_bytes()
