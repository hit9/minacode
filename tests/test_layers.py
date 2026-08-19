"""Dependency-direction guard: module-level imports must follow the module layers.

Layers, from highest to lowest:

    __main__  ->  cli/  ->  tui.py / render.py
                   |
               engine.py
                   |
         context.py / runner.py
                   |
                model.py
                   |
       tools/   mcp.py   skill.py
                   |
               session/
                   |
                image.py
                   |
        base.py  config.py  provider_compat.py
                   |
             model_catalog.py

Same-layer imports are allowed; cross-layer imports may only point downward.
prompts.py, hints.py and update.py are leaves: any layer may import them, and their own
imports are unconstrained. TYPE_CHECKING blocks and function-local imports are not counted.
"""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MINACODE = REPO / "minacode"

# Module prefix -> layer number (larger = lower). Same-layer allowed; a source may only import
# targets whose layer is >= its own.
LAYERS = {
    "minacode.__init__": 0,
    "minacode.__main__": 0,
    "minacode.cli": 1,
    "minacode.tui": 2,
    "minacode.render": 2,
    "minacode.engine": 3,
    "minacode.context": 4,
    "minacode.runner": 4,
    "minacode.model": 5,
    "minacode.model.chat": 5,
    "minacode.model.responses": 5,
    "minacode.model.anthropic": 5,
    "minacode.model.client": 5,
    "minacode.model.resilience": 5,
    "minacode.tools": 6,
    "minacode.mcp": 6,
    "minacode.skill": 6,
    "minacode.mentions": 6,
    "minacode.builtin_skills": 6,
    "minacode.session": 7,
    "minacode.image": 8,
    "minacode.base": 9,
    "minacode.config": 9,
    "minacode.provider_compat": 9,
    "minacode.model_catalog": 10,
}
# Leaves: any layer may depend on them; their own dependencies are not checked.
LEAVES = ("minacode.prompts", "minacode.hints", "minacode.update")


def layer_of(module: str) -> int | None:
    """The layer for a module, matched by longest prefix."""
    best = None
    for prefix, layer in LAYERS.items():
        if module == prefix or module.startswith(prefix + "."):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, layer)
    return best[1] if best else None


def module_level_minacode_imports(path: pathlib.Path) -> list[str]:
    """Module-level `import minacode.x` / `from minacode.x import ...` targets."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and isinstance(node._parent, ast.Module):
            if node.module and node.module.startswith("minacode"):
                targets.append(node.module)
        elif isinstance(node, ast.Import) and isinstance(node._parent, ast.Module):
            for alias in node.names:
                if alias.name.startswith("minacode"):
                    targets.append(alias.name)
    return targets


def _annotate(tree: ast.AST) -> ast.AST:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent
    return tree


def all_sources() -> dict[str, pathlib.Path]:
    return {
        "minacode." + path.relative_to(MINACODE).as_posix()[:-3].replace("/", "."): path
        for path in MINACODE.rglob("*.py")
    }


def source_layer(module: str) -> int:
    layer = layer_of(module)
    assert layer is not None, f"no layer configured for {module}"
    return layer


@pytest.mark.parametrize("module", sorted(all_sources()))
def test_module_level_imports_stay_within_or_below_layer(module: str):
    path = all_sources()[module]
    if any(module == leaf or module.startswith(leaf + ".") for leaf in LEAVES):
        return  # leaves may import anything
    source = source_layer(module)
    for target in module_level_minacode_imports(path):
        if any(target == leaf or target.startswith(leaf + ".") for leaf in LEAVES):
            continue
        target_layer = layer_of(target)
        assert target_layer is not None, f"{module} imports {target}, which has no configured layer"
        assert target_layer >= source, (
            f"{path} imports {target} ({target} is layer {target_layer}, above {module}'s layer {source}); "
            "cross-layer imports may only point downward"
        )


def test_every_subpackage_is_declared_for_distribution():
    """setuptools uses an explicit package list, so a new subpackage that nobody adds there is
    simply absent from the wheel. Tests run from the source tree and never notice; the failure
    surfaces only as an ImportError after install, against whatever stale files remain."""
    import tomllib

    declared = set(tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    found = {
        "minacode." + path.parent.relative_to(MINACODE).as_posix().replace("/", ".")
        for path in MINACODE.rglob("__init__.py")
        if path.parent != MINACODE
    } | {"minacode"}
    assert found == declared, f"pyproject packages out of sync: missing {sorted(found - declared)}, stale {sorted(declared - found)}"
