"""Offline tests for the pure parsing/cleaning helpers.

Needs neither ComfyUI nor llama-cpp-python.  Run with:
    python tests/test_parsing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Story_Prompts_Node.story_prompt_node as node


def test_parse_character_map():
    entries = node.parse_character_map(
        "# comment line\n"
        "Mira | mira_char_v2.safetensors | mira_xj23 | a young woman with short auburn hair\n"
        "Old Mill | | | a crumbling stone watermill beside a fast river\n"
        "Jonas\n"
        " | bad line with no name\n"
        "Mira | duplicate | should | be skipped\n"
    )
    assert [e["name"] for e in entries] == ["Mira", "Old Mill", "Jonas"]
    assert entries[0]["lora"] == "mira_char_v2.safetensors"
    assert entries[0]["trigger"] == "mira_xj23"
    assert entries[1]["lora"] == "" and entries[1]["trigger"] == ""
    assert entries[1]["description"].startswith("a crumbling")
    assert entries[2] == {"name": "Jonas", "lora": "", "trigger": "", "description": ""}


def test_bible_parse_and_merge():
    map_entries = node.parse_character_map(
        "Mira | lora.safetensors | mira_xj23 | a young woman with short auburn hair\n"
        "Jonas\n"
    )
    llm_entries = node.parse_bible_reply(
        "Here are the entries:\n"
        "Mira: a tall woman in red\n"
        "**Jonas**: a wiry old ferryman with a silver beard\n"
        "- The River: dark, fast water under mist\n"
        "This line has no colon and is ignored\n"
    )
    assert ("Mira", "a tall woman in red") in llm_entries
    bible = node.merge_bible(map_entries, llm_entries)
    by = {e["name"]: e for e in bible}
    assert by["Mira"]["description"].startswith("a young woman")   # user's map wins
    assert by["Mira"]["trigger"] == "mira_xj23"
    assert by["Jonas"]["description"].startswith("a wiry old")     # LLM fills the blank
    assert "The River" in by                                       # extra LLM entry appended
    assert by["The River"]["trigger"] == ""
    text = node.format_bible(bible)
    assert "trigger: mira_xj23" in text and "The River:" in text


def test_parse_scene_reply():
    scenes = node.parse_scene_reply(
        "SCENE 1: Mira crosses the rope bridge at dawn, clutching the brass key.\n"
        "CHARACTERS: Mira, none\n"
        "\n"
        "**SCENE 2:** Jonas poles the ferry through fog near the Old Mill.\n"
        "CHARACTERS: Jonas\n",
        ["Mira", "Jonas", "Old Mill"], 5)
    assert len(scenes) == 2
    assert scenes[0]["characters"] == ["Mira"]
    assert "rope bridge" in scenes[0]["synopsis"]
    # Old Mill picked up from the synopsis even though the CHARACTERS line missed it
    assert set(scenes[1]["characters"]) == {"Jonas", "Old Mill"}
    # respects the cap
    assert len(node.parse_scene_reply("SCENE 1: a\nSCENE 2: b\nSCENE 3: c", [], 2)) == 2
    # unparseable input -> empty, not a crash
    assert node.parse_scene_reply("no scenes here at all", ["Mira"], 3) == []


def test_clean_prompt():
    cleaned = node.clean_prompt(
        "Here is the prompt:\n"
        "**Subject:** A young woman stands at the mill door. Lighting: golden hour rays "
        "spill across the floor. The mood is calm.",
        ["mira_xj23"])
    assert "Subject" not in cleaned
    assert "Lighting:" not in cleaned
    assert "**" not in cleaned
    assert "\n" not in cleaned
    assert cleaned.lower().startswith("mira_xj23, ")               # missing trigger prepended
    assert "mira_xj23" in cleaned                                  # underscores preserved

    # present trigger is not duplicated
    cleaned2 = node.clean_prompt("mira_xj23, a woman walks along the river.", ["mira_xj23"])
    assert cleaned2.count("mira_xj23") == 1

    # numbered-list prefix and surrounding quotes stripped
    cleaned3 = node.clean_prompt('1. "A quiet harbor at dusk, lanterns glowing."')
    assert cleaned3.startswith("A quiet harbor")
    assert not cleaned3.startswith('"')


def test_split_paragraph_chunks():
    text = "\n\n".join(f"Paragraph {i} " + ("x" * 80) for i in range(10))
    chunks = node.split_paragraph_chunks(text, 300)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(c.count("Paragraph") for c in chunks) == 10         # nothing lost

    big = ("Sentence one. " * 100).strip()                         # one huge paragraph
    chunks2 = node.split_paragraph_chunks(big, 200)
    assert all(len(c) <= 200 for c in chunks2)
    assert sum(c.count("Sentence") for c in chunks2) == 100


def test_style_presets():
    real = [k for k, v in node.STYLE_PRESETS.items() if v]
    assert len(real) >= 10
    assert any(k.startswith("SFW") for k in real)
    assert any(k.startswith("NSFW") for k in real)
    assert node.resolve_style(node.STYLE_NONE) == ""                       # infer per scene
    assert node.resolve_style(node.STYLE_CUSTOM, " oil pastel sketch ") == "oil pastel sketch"
    assert node.resolve_style(node.STYLE_CUSTOM, "") == ""
    assert node.resolve_style("SFW - anime cel shading") != ""
    assert node.resolve_style("freeform typed style") == "freeform typed style"  # API passthrough
    # the pre-filled map template is all comments -> parses to an empty map
    assert node.parse_character_map(node.CHARACTER_MAP_TEMPLATE) == []


def test_strip_reasoning():
    from Story_Prompts_Node.llm_engine import strip_reasoning
    assert strip_reasoning("<think>plan the shot list</think>The final prompt.") == "The final prompt."
    # some chat templates auto-open the block, so only a closing tag appears
    assert strip_reasoning("reasoning with no opener</think>Answer here.") == "Answer here."
    # reasoning truncated by max_tokens: nothing usable remains
    assert strip_reasoning("<think>truncated reasoning that never closes") == ""
    assert strip_reasoning("No tags at all.") == "No tags at all."
    assert strip_reasoning("<think>a</think>draft<think>b</think>Final.") == "Final."


def test_node_class_shape():
    cls = node.StoryPromptBatchGenerator
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == len(cls.OUTPUT_IS_LIST) == 4
    assert cls.OUTPUT_IS_LIST == (True, False, False, True)
    spec = cls.INPUT_TYPES()                                       # works without ComfyUI
    required = spec["required"]
    for widget in ("story_file", "gguf_model", "num_prompts", "context_length", "seed",
                   "style_medium", "character_lora_map", "system_instructions",
                   "output_dir", "filename_prefix", "enable"):
        assert widget in required, widget
    assert required["system_instructions"][1]["default"] == node.KREA2_SYSTEM_INSTRUCTIONS
    assert "[Subject] + [Action/Pose]" in node.KREA2_SYSTEM_INSTRUCTIONS
    assert cls.VALIDATE_INPUTS(story_file=node.NO_STORY_PLACEHOLDER) is not True
    assert node._sanitize_prefix("../evil name!") == "evil_name"
    assert node._sanitize_prefix("") == "scene"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
