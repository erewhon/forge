"""Editor.js block builder helpers."""

from forge.shared.models.nous import EditorJsBlock


def header(text: str, level: int = 2) -> EditorJsBlock:
    return EditorJsBlock(type="header", data={"text": text, "level": level})


def paragraph(text: str) -> EditorJsBlock:
    return EditorJsBlock(type="paragraph", data={"text": text})


def _nest_items(items: list[str]) -> list[dict]:
    return [{"content": item, "items": []} for item in items]


def unordered_list(items: list[str]) -> EditorJsBlock:
    return EditorJsBlock(type="list", data={"items": _nest_items(items), "style": "unordered"})


def ordered_list(items: list[str]) -> EditorJsBlock:
    return EditorJsBlock(type="list", data={"items": _nest_items(items), "style": "ordered"})


def checklist(items: list[str], checked: bool = False) -> EditorJsBlock:
    return EditorJsBlock(
        type="checklist",
        data={"items": [{"text": item, "checked": checked} for item in items]},
    )
