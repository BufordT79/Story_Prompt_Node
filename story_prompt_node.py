"""Story → Krea 2 Prompt Batch generator node.

Reads a story from a .txt file in ComfyUI/input, uses a local GGUF LLM
(llama-cpp-python) to pick the most visually compelling moments, and writes one
Krea 2-formatted natural-language image prompt per moment — emitted as an
in-graph STRING list and as numbered .txt files on disk.

Generation runs in three passes:
  1. bible pass      — extract/merge consistent character & setting descriptions
  2. scene pass      — select up to num_prompts key visual moments
  3. prompt pass     — write one Krea 2 prose prompt per selected scene
Stories that exceed the context budget are first compressed with a
chunk-and-summarize map-reduce.
"""

import hashlib
import os
import re

try:
    import folder_paths
except ImportError:  # allows importing this module outside ComfyUI (tests)
    folder_paths = None

from .llm_engine import chat, count_tokens, get_model

_LOG_PREFIX = "[StoryPromptBatch]"

NO_STORY_PLACEHOLDER = "(none found - put .txt files in ComfyUI/input)"
NO_MODEL_PLACEHOLDER = "(none found - put .gguf models in ComfyUI/models/LLM)"

STYLE_NONE = "none - infer per scene from the story"
STYLE_CUSTOM = "custom - use the style_custom field"

# dropdown label -> style text handed to the LLM ("" = infer, None = use style_custom)
STYLE_PRESETS = {
    STYLE_NONE: "",
    "SFW - cinematic film still": "a cinematic film still with shallow depth of field and rich color grading",
    "SFW - moody atmospheric photography": "moody atmospheric photography with dramatic natural light",
    "SFW - golden hour portrait photography": "golden hour portrait photography with warm backlight",
    "SFW - 35mm analog film photo": "35mm analog film photography with subtle grain and muted tones",
    "SFW - watercolor storybook illustration": "a soft watercolor storybook illustration",
    "SFW - classical oil painting": "a classical oil painting with visible brushwork",
    "SFW - anime cel shading": "anime-style cel shading with clean line art",
    "SFW - dark fantasy digital painting": "a dark fantasy digital painting, detailed and painterly",
    "SFW - comic book graphic novel art": "bold comic book graphic novel art with inked outlines",
    "SFW - cinematic 3D render": "a cinematic 3D render with physically based lighting",
    "NSFW - boudoir photography": "intimate boudoir photography with soft warm lighting",
    "NSFW - sensual glamour photography": "sensual editorial glamour photography",
    "NSFW - artistic nude photography": "artistic nude photography with chiaroscuro lighting",
    "NSFW - erotic digital painting": "an erotic digital painting, detailed and painterly",
    "NSFW - hentai anime style": "explicit hentai-style anime art",
    STYLE_CUSTOM: None,
}


def resolve_style(style_medium, style_custom=""):
    """Map a style_medium dropdown label to the style text given to the LLM.

    Unknown labels pass through unchanged so API-built workflows can still
    supply free text directly.
    """
    text = STYLE_PRESETS.get(style_medium, style_medium)
    if text is None:  # the custom option
        text = style_custom or ""
    return (text or "").strip()


CHARACTER_MAP_TEMPLATE = """\
# One entry per line:  Name | lora_file (optional) | trigger word (optional) | visual description (optional)
# Delete the leading # to activate a line. Detailed examples:
# Lena | lena_flux_v3.safetensors | lena_v3 | a curvy young woman in her mid-20s with long platinum-blonde hair, blue eyes, a small star tattoo on her left collarbone, and a fitted black slip dress
# Marcus | | | a tall broad-shouldered man with a trimmed dark beard, a buzz cut, and a gray henley with rolled-up sleeves
# Neon Loft | loft_env_v1.safetensors | neonloft | a downtown loft at night with exposed brick, pink-and-blue neon signs, and rain-streaked floor-to-ceiling windows
"""

KREA2_SYSTEM_INSTRUCTIONS = """You are a specialized prompt engineer for the Krea 2 image generation model. You turn the
scene you are given — a story moment plus any required character/setting descriptions — into
ONE polished, vivid, natural-language prose prompt. You upgrade and clarify WITHOUT changing
what the scene is going for.

PRIORITY RULES (obey in this order — top is most important)
1. EXPLICIT REQUIREMENTS FIRST. Any specific requirement given with the scene (a required
   character or setting description, a trigger token to weave in, a required style/medium) is
   the top priority — apply it exactly. Everything below only complements it; never contradict
   or drop it.
2. SCENE AS BASE. Keep the scene's meaning and every key element — subject, action, setting.
   Do NOT swap them for something different. Preserve the story's intended tone and style; if
   none is implied or required, default to a natural, photorealistic look — do not impose an
   aesthetic the scene did not ask for.
3. CONTENT / NSFW. Keep the content exactly as the scene implies — do not add or remove nudity
   or intimacy on your own. When explicit content is present, describe it plainly with precise
   anatomical terms — never euphemize, censor, refuse, or add disclaimers.
4. COHERENCE. The prompt must be internally consistent. If details conflict (lighting, camera,
   setting, clothing state...), resolve in this order: explicit requirements first, then the
   scene's own intent, then defaults — keep the higher-priority element and drop or adapt the
   lower one. Never output contradictory details.

HOW TO WRITE THE PROMPT
Krea 2 responds best to long, descriptive, coherent natural-language prose. It relies on
context rather than mechanical keyword stuffing — NEVER write comma-separated tags or keyword
lists. Write flowing, connected sentences that naturally cover, in this order:

[Subject] + [Action/Pose] + [Scene/Location] + [Lighting] + [Color] +
[Composition & Camera] + [Style/Medium]

- Add the missing but implied specifics: lighting, framing, camera feel, textures, mood —
  enough to guide the model, not to overwrite intent.
- Turn vague words into concrete visuals ("pretty" becomes the actual features).
- Stay believable and coherent; don't pile on contradictory details.

OUTPUT
ONE continuous flowing English prose paragraph — natural sentences, no line breaks, no labels
(no "Subject:"), no headers, no commentary, no reasoning, no think blocks. Do not use negative
prompts; Krea 2 does not use them. Aim for roughly 75-100 words.

EXAMPLE
Scene: a young woman drinking coffee by a cafe window in the morning, casual, realistic.
Prompt: A young woman sits by a large cafe window in soft daylight, both hands wrapped around
a warm ceramic cup, wearing a relaxed oversized knit sweater, hair loosely tucked behind one
ear. Gentle window light falls across her face and the wooden table, catching the steam rising
from the coffee and fine dust in the air, the blurred cafe interior warm behind her. Natural
candid composition, realistic skin and textures, calm everyday mood, a softly lit
photorealistic portrait."""


# --------------------------------------------------------------------------
# file discovery (ComfyUI/input stories, ComfyUI/models/LLM gguf models)
# --------------------------------------------------------------------------

def _input_dir():
    if folder_paths is not None:
        return folder_paths.get_input_directory()
    return os.path.join(os.getcwd(), "input")


def _list_story_files():
    base = _input_dir()
    found = []
    if os.path.isdir(base):
        for root, _dirs, names in os.walk(base):
            for name in names:
                if name.lower().endswith(".txt"):
                    rel = os.path.relpath(os.path.join(root, name), base)
                    found.append(rel.replace("\\", "/"))
    return sorted(found, key=str.lower)


def _resolve_story_path(story_file):
    base = os.path.abspath(_input_dir())
    path = os.path.abspath(os.path.join(base, story_file))
    if path != base and not path.startswith(base + os.sep):
        return None  # refuse paths that escape the input directory
    return path


def _register_llm_folder():
    if folder_paths is None:
        return
    llm_dir = os.path.join(folder_paths.models_dir, "LLM")
    os.makedirs(llm_dir, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("LLM", llm_dir)
        entry = folder_paths.folder_names_and_paths.get("LLM")
        if entry is not None and isinstance(entry[1], set):
            entry[1].add(".gguf")
    except Exception as exc:
        print(f"{_LOG_PREFIX} could not register models/LLM with folder_paths: {exc}")


def _list_gguf_models():
    if folder_paths is not None:
        _register_llm_folder()
        try:
            names = [n for n in folder_paths.get_filename_list("LLM")
                     if n.lower().endswith(".gguf")]
            if names:
                return names
        except Exception:
            pass
        base = os.path.join(folder_paths.models_dir, "LLM")
    else:
        base = os.path.join(os.getcwd(), "models", "LLM")
    found = []
    if os.path.isdir(base):
        for root, _dirs, names in os.walk(base):
            for name in names:
                if name.lower().endswith(".gguf"):
                    rel = os.path.relpath(os.path.join(root, name), base)
                    found.append(rel.replace("\\", "/"))
    return sorted(found, key=str.lower)


def _resolve_model_path(name):
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("LLM", name)
            if path:
                return path
        except Exception:
            pass
        return os.path.join(folder_paths.models_dir, "LLM", name)
    return os.path.join(os.getcwd(), "models", "LLM", name)


def _read_story(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read().strip()


# --------------------------------------------------------------------------
# pure parsing helpers (unit-tested in tests/test_parsing.py)
# --------------------------------------------------------------------------

def parse_character_map(text):
    """Parse the pipe-delimited character_lora_map into entry dicts.

    Line format:  Name | lora_file | trigger word | visual description
    Everything after Name is optional; malformed lines are skipped with a
    warning, never a crash. '#' starts a comment line.
    """
    entries = []
    seen = set()
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        if not name:
            print(f"{_LOG_PREFIX} skipping malformed character_lora_map line (no name): {raw_line!r}")
            continue
        if name.casefold() in seen:
            print(f"{_LOG_PREFIX} skipping duplicate character_lora_map entry: {name!r}")
            continue
        seen.add(name.casefold())
        entries.append({
            "name": name,
            "lora": parts[1] if len(parts) > 1 else "",
            "trigger": parts[2] if len(parts) > 2 else "",
            "description": " | ".join(parts[3:]).strip() if len(parts) > 3 else "",
        })
    return entries


_BIBLE_LINE = re.compile(r"^\s*[-*\d.)\s]*([^:\n]{1,60}?)\s*:\s*(.+?)\s*$")


def parse_bible_reply(text):
    """Parse 'NAME: description' lines out of the bible-pass LLM reply."""
    entries = []
    seen = set()
    for line in (text or "").splitlines():
        match = _BIBLE_LINE.match(line)
        if not match:
            continue
        name = match.group(1).strip().strip("*_`\"'")
        desc = match.group(2).strip().strip("*`")
        if not name or not desc or len(name.split()) > 6:
            continue
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        entries.append((name, desc))
    return entries


def merge_bible(map_entries, llm_entries):
    """Merge user map entries with LLM-extracted ones; the user's map wins."""
    merged = []
    by_name = {}
    for entry in map_entries:
        item = dict(entry)
        merged.append(item)
        by_name[item["name"].casefold()] = item
    for name, desc in llm_entries:
        key = name.casefold()
        if key in by_name:
            if not by_name[key]["description"]:
                by_name[key]["description"] = desc
        else:
            item = {"name": name, "lora": "", "trigger": "", "description": desc}
            merged.append(item)
            by_name[key] = item
    return merged


def format_bible(bible):
    lines = []
    for entry in bible:
        line = f"{entry['name']}: {entry['description'] or '(no description)'}"
        extras = []
        if entry["trigger"]:
            extras.append(f"trigger: {entry['trigger']}")
        if entry["lora"]:
            extras.append(f"lora: {entry['lora']}")
        if extras:
            line += f"  [{', '.join(extras)}]"
        lines.append(line)
    return "\n".join(lines)


_SCENE_HEAD = re.compile(r"(?im)^[ \t]*[#>*\-\d.\s]*scene\s*(\d+)\s*[:.\-–—]\s*")
_CHAR_LINE = re.compile(r"(?im)^\s*[*\-\s]*characters?\s*[:.\-]\s*(.*)$")


def parse_scene_reply(text, known_names, max_scenes):
    """Parse SCENE N / CHARACTERS blocks from the scene-selection reply.

    Characters are taken from the CHARACTERS line when it matches a known
    name, plus a fuzzy union with known names mentioned in the synopsis (so a
    forgotten CHARACTERS line doesn't lose the consistency injection).
    """
    text = text or ""
    heads = list(_SCENE_HEAD.finditer(text))
    lookup = {n.casefold(): n for n in known_names}
    scenes = []
    for idx, head in enumerate(heads):
        start = head.end()
        end = heads[idx + 1].start() if idx + 1 < len(heads) else len(text)
        block = text[start:end]
        char_match = _CHAR_LINE.search(block)
        if char_match:
            synopsis_raw = block[:char_match.start()]
            char_raw = char_match.group(1)
        else:
            synopsis_raw = block
            char_raw = ""
        synopsis = re.sub(r"\s+", " ", synopsis_raw).strip(" \t\r\n*-")
        if not synopsis:
            continue
        characters = []
        for candidate in char_raw.split(","):
            hit = lookup.get(candidate.strip().strip("*_`\"'").casefold())
            if hit and hit not in characters:
                characters.append(hit)
        for name in known_names:
            if name not in characters and re.search(rf"(?i)\b{re.escape(name)}\b", synopsis):
                characters.append(name)
        scenes.append({"synopsis": synopsis, "characters": characters})
        if len(scenes) >= max_scenes:
            break
    return scenes


_LABEL = re.compile(
    r"(?i)[\[({]?\b(subject|action(?:\s*/\s*pose)?|pose|scene(?:\s*/\s*location)?|location|"
    r"lighting|colou?rs?|composition(?:\s*(?:&|and)\s*camera)?|camera|style(?:\s*/\s*medium)?|medium)"
    r"\b[\])}]?\s*:\s*"
)
_PREAMBLE = re.compile(r"(?i)^(here(?:'s| is| are)|sure|okay|of course|certainly|below)\b.*$")


def clean_prompt(text, triggers=()):
    """Normalize an LLM reply into one clean prose prompt.

    Strips code fences, chatty preambles, leaked formula labels ('Subject:',
    'Lighting:', ...), and markdown; collapses to a single paragraph; and
    guarantees every required LoRA trigger word is present (prepended if the
    model dropped it). Underscores are preserved — trigger words depend on them.
    """
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0].strip()
        if _PREAMBLE.match(first) or (first.endswith(":") and len(first) < 60):
            lines = lines[1:]
    text = " ".join(lines)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    text = _LABEL.sub(" ", text)
    text = re.sub(r"[*#`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"“” ').strip()
    for trigger in triggers:
        if trigger and not re.search(re.escape(trigger), text, re.IGNORECASE):
            print(f"{_LOG_PREFIX} trigger {trigger!r} was missing from the prompt; prepending it")
            text = f"{trigger}, {text}"
    return text


def split_paragraph_chunks(text, max_chars):
    """Split text into chunks of at most max_chars, preferring paragraph
    boundaries, then sentence boundaries for oversized single paragraphs."""
    max_chars = max(200, int(max_chars))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut >= max_chars // 4 else max_chars
            flush()
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            flush()
        current = f"{current}\n\n{para}" if current else para
    flush()
    return chunks


# --------------------------------------------------------------------------
# ComfyUI runtime helpers
# --------------------------------------------------------------------------

def _check_interrupted():
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def _make_pbar(total):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        return None


def _resolve_output_dir(output_dir):
    path = (output_dir or "").strip() or "output/story_prompts"
    if not os.path.isabs(path):
        root = getattr(folder_paths, "base_path", None) if folder_paths else None
        path = os.path.join(root or os.getcwd(), path)
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_prefix(prefix):
    prefix = re.sub(r"[^A-Za-z0-9_\-]+", "_", (prefix or "").strip()).strip("_")
    return prefix or "scene"


def _write_prompt_files(prompts, output_dir, prefix):
    pad = max(2, len(str(len(prompts))))
    paths = []
    for i, prompt in enumerate(prompts, start=1):
        path = os.path.join(output_dir, f"{prefix}_{i:0{pad}d}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(prompt + "\n")
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# LLM pass message builders
# --------------------------------------------------------------------------

def _bible_messages(story, map_entries):
    known = ""
    if map_entries:
        lines = [f"- {e['name']}: {e['description'] or '(write a description)'}"
                 for e in map_entries]
        known = (
            "The user has already defined these characters/settings. Keep their names exactly "
            "as written. Where a description is given, repeat it verbatim; where it says "
            "(write a description), write one from the story:\n" + "\n".join(lines) + "\n\n"
        )
    user = (
        "Read the story below. Identify the recurring characters and the important "
        "settings/locations that appear in more than one moment of the story.\n\n" + known +
        "For each, write ONE consistent visual description in under 40 words (appearance, "
        "clothing and distinctive features for characters; architecture, atmosphere and "
        "distinctive features for settings), written so the exact same wording can be reused "
        "in every image prompt where they appear.\n\n"
        "Reply with one line per entry, in exactly this format, and nothing else:\n"
        "NAME: description\n\n"
        "STORY:\n" + story
    )
    return [
        {"role": "system", "content": "You analyze stories to keep characters and settings "
                                      "visually consistent across illustrations."},
        {"role": "user", "content": user},
    ]


def _scene_messages(story, bible, num_prompts):
    known = ""
    if bible:
        known = ("Known characters/settings — when one appears in a scene, use its exact name:\n"
                 + "\n".join(f"- {e['name']}" for e in bible) + "\n\n")
    user = (
        f"Read the story below and select the {num_prompts} most visually compelling key "
        "moments to illustrate, in chronological story order, spread across the whole story. "
        "Each moment must be one concrete, visually depictable scene. If the story genuinely "
        f"contains fewer than {num_prompts} distinct visual moments, select only as many as it "
        "truly supports — never invent events that are not in the story.\n\n" + known +
        "Reply with one block per scene, in exactly this format, and nothing else:\n"
        "SCENE 1: two or three sentence synopsis of the moment, concrete and visual\n"
        "CHARACTERS: comma-separated names from the known list that appear in it, or none\n\n"
        "STORY:\n" + story
    )
    return [
        {"role": "system", "content": "You select key visual moments from stories for illustration."},
        {"role": "user", "content": user},
    ]


def _prompt_messages(system_instructions, scene, bible_by_name, style_medium):
    parts = [f"Scene from the story:\n{scene['synopsis']}"]
    consistency = []
    for name in scene["characters"]:
        entry = bible_by_name.get(name.casefold())
        if entry is None:
            continue
        line = f"- {entry['name']}: {entry['description']}" if entry["description"] else f"- {entry['name']}"
        if entry["trigger"]:
            line += (f' (weave the exact token "{entry["trigger"]}" naturally into the prose where '
                     f'{entry["name"]} is first described — for example "{entry["trigger"]}, '
                     f'{entry["description"] or "..."}" — never append it as a tag at the end)')
        consistency.append(line)
    if consistency:
        parts.append(
            "Characters/settings in this scene — describe each using EXACTLY these visual "
            "descriptions, for consistency with the other images in this batch:\n"
            + "\n".join(consistency)
        )
    if style_medium.strip():
        parts.append(f"Required style/medium for this image: {style_medium.strip()}")
    parts.append(
        "Write ONE Krea 2 image prompt for this scene now. Output only the prompt text itself — "
        "no title, no labels, no quotes, no commentary."
    )
    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _capped_max_tokens(model, messages, want, n_ctx):
    """Clamp a generation budget so prompt + output always fit in n_ctx."""
    used = sum(count_tokens(model, m["content"]) for m in messages) + 64
    room = n_ctx - used - 16
    if room < 64:
        raise RuntimeError(
            f"The LLM input needs ~{used} tokens but context_length is only {n_ctx}. "
            "Increase context_length, lower num_prompts, or use a shorter story."
        )
    return max(64, min(int(want), room))


def _compress_story(model, story, budget_tokens, n_ctx, seed, log):
    """Chunk-and-summarize map-reduce until the story fits the context budget."""
    rounds = 0
    while count_tokens(model, story) > budget_tokens and rounds < 3:
        rounds += 1
        chunk_chars = max(2000, int(budget_tokens * 0.6) * 4)
        chunks = split_paragraph_chunks(story, chunk_chars)
        per_chunk_out = max(150, min(700, budget_tokens // max(1, len(chunks))))
        log(f"story exceeds the context budget ({budget_tokens} tokens); compression round "
            f"{rounds}: {len(chunks)} chunks -> ~{per_chunk_out} tokens each")
        summaries = []
        for i, chunk in enumerate(chunks):
            _check_interrupted()
            messages = [
                {"role": "system", "content": "You condense stories without losing visual detail."},
                {"role": "user", "content":
                    f"Condense this story excerpt to at most {per_chunk_out * 3 // 4} words. "
                    "Preserve every character name, the sequence of events, and all key visual "
                    "details — appearances, settings, lighting, atmosphere. Write plain prose, "
                    "no headings.\n\nEXCERPT:\n" + chunk},
            ]
            max_out = _capped_max_tokens(model, messages, per_chunk_out, n_ctx)
            summaries.append(chat(model, messages, max_tokens=max_out, seed=seed + 9000 + i,
                                  temperature=0.3, top_p=0.9, top_k=40, repeat_penalty=1.1))
        story = "\n\n".join(summaries)
    if count_tokens(model, story) > budget_tokens:
        log("story still exceeds the budget after 3 compression rounds; hard-truncating")
        story = story[: budget_tokens * 4]
    return story


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class StoryPromptBatchGenerator:
    """Turn a story .txt into a batch of Krea 2 image prompts via a local GGUF LLM."""

    _LAST_RESULT = None

    CATEGORY = "prompt/story"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "prompt_count", "character_bible", "saved_paths")
    OUTPUT_IS_LIST = (True, False, False, True)
    OUTPUT_TOOLTIPS = (
        "Generated Krea 2 prompts in story order (a list — downstream nodes run once per prompt)",
        "Actual number of prompts generated (may be fewer than requested; never padded)",
        "The resolved character/setting descriptions used across the batch",
        "Disk paths of the written prompt .txt files",
    )
    DESCRIPTION = (
        "Reads a story .txt from ComfyUI/input, picks its key visual moments with a local GGUF "
        "LLM, and writes one long-form natural-language Krea 2 prompt per moment — with "
        "consistent recurring characters/settings and optional LoRA trigger words woven in. "
        "Prompts are returned as a list and saved as numbered .txt files."
    )

    @classmethod
    def INPUT_TYPES(cls):
        stories = _list_story_files() or [NO_STORY_PLACEHOLDER]
        models = _list_gguf_models() or [NO_MODEL_PLACEHOLDER]
        return {
            "required": {
                "story_file": (stories, {"tooltip": "Story .txt file, read from ComfyUI/input "
                                                    "(subfolders included). Press R to refresh."}),
                "gguf_model": (models, {"tooltip": "Local GGUF language model from "
                                                   "ComfyUI/models/LLM. Use an instruct-tuned model."}),
                "num_prompts": ("INT", {"default": 8, "min": 1, "max": 200,
                                        "tooltip": "How many image prompts to generate. If the story "
                                                   "supports fewer distinct moments, you get fewer — "
                                                   "never filler."}),
                "context_length": ("INT", {"default": 8192, "min": 512, "max": 131072, "step": 512,
                                           "tooltip": "LLM context window (n_ctx). Longer stories and "
                                                      "larger num_prompts need more."}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200,
                                       "tooltip": "Layers offloaded to GPU: -1 = all (needs a "
                                                  "CUDA/Metal build of llama-cpp-python), 0 = CPU only."}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01}),
                "max_tokens_per_prompt": ("INT", {"default": 200, "min": 32, "max": 4096,
                                                  "tooltip": "Generation cap per prompt (~2-5 "
                                                             "sentences at the default). Thinking "
                                                             "models need much more (600+): their "
                                                             "reasoning spends from this budget "
                                                             "before the prompt appears."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "tooltip": "Sampling seed. Fix it to reproduce a batch; each scene "
                                            "derives its own sub-seed."}),
                "style_medium": (list(STYLE_PRESETS.keys()),
                                 {"default": STYLE_NONE,
                                  "tooltip": "Style/Medium preset forced onto every prompt in the "
                                             "batch. 'none' lets the LLM infer it per scene; "
                                             "'custom' uses the style_custom field. NSFW presets "
                                             "pair with an uncensored GGUF."}),
                "character_lora_map": ("STRING", {"default": CHARACTER_MAP_TEMPLATE,
                                                  "multiline": True,
                                                  "tooltip": "One entry per line:\nName | lora_file | "
                                                             "trigger word | visual description\n"
                                                             "Everything after Name is optional; "
                                                             "# starts a comment. The pre-filled "
                                                             "lines are a template — delete the # "
                                                             "to activate one."}),
                "system_instructions": ("STRING", {"default": KREA2_SYSTEM_INSTRUCTIONS,
                                                   "multiline": True,
                                                   "tooltip": "The Krea 2 prompt-engineering rules "
                                                              "given to the LLM. Editable."}),
                "output_dir": ("STRING", {"default": "output/story_prompts",
                                          "tooltip": "Where prompt .txt files are written. Relative "
                                                     "paths resolve against the ComfyUI folder. "
                                                     "Reruns overwrite."}),
                "filename_prefix": ("STRING", {"default": "scene"}),
                "enable": ("BOOLEAN", {"default": True,
                                       "tooltip": "Off = skip the LLM and return the previous batch "
                                                  "(or empty outputs)."}),
            },
            "optional": {
                "style_custom": ("STRING", {"default": "",
                                            "tooltip": "Free-text style/medium — used only when "
                                                       "style_medium is set to the custom option."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, story_file=NO_STORY_PLACEHOLDER, **kwargs):
        path = _resolve_story_path(story_file) if story_file else None
        if path and os.path.isfile(path):
            digest = hashlib.sha256()
            with open(path, "rb") as f:
                digest.update(f.read())
            return digest.hexdigest()
        return ""

    @classmethod
    def VALIDATE_INPUTS(cls, story_file=None, gguf_model=None):
        if story_file is not None:
            if story_file == NO_STORY_PLACEHOLDER:
                return ("No story selected: put a .txt file in ComfyUI/input and press R "
                        "to refresh the node.")
            path = _resolve_story_path(story_file)
            if path is None or not os.path.isfile(path):
                return f"Story file not found in ComfyUI/input: {story_file}"
        if gguf_model is not None and gguf_model == NO_MODEL_PLACEHOLDER:
            return ("No GGUF model selected: put a .gguf file in ComfyUI/models/LLM and press R "
                    "to refresh the node.")
        return True

    def generate(self, story_file, gguf_model, num_prompts, context_length, gpu_layers,
                 temperature, top_p, top_k, repeat_penalty, max_tokens_per_prompt, seed,
                 style_medium, character_lora_map, system_instructions,
                 output_dir, filename_prefix, enable, style_custom=""):

        def log(msg):
            print(f"{_LOG_PREFIX} {msg}")

        if not enable:
            cached = type(self)._LAST_RESULT
            if cached is not None:
                log("disabled — returning the previous batch")
                return cached
            log("disabled — returning empty outputs")
            return {"ui": {"text": ["(disabled)"]}, "result": ([], 0, "", [])}

        # ---- inputs
        if story_file == NO_STORY_PLACEHOLDER:
            raise RuntimeError("No story file: put a .txt file in ComfyUI/input and press R "
                               "to refresh the node.")
        story_path = _resolve_story_path(story_file)
        if story_path is None or not os.path.isfile(story_path):
            raise RuntimeError(f"Story file not found in ComfyUI/input: {story_file}")
        story = _read_story(story_path)
        if not story:
            raise RuntimeError(f"Story file is empty: {story_path}")

        if gguf_model == NO_MODEL_PLACEHOLDER:
            raise RuntimeError("No GGUF model: put a .gguf file in ComfyUI/models/LLM and press R "
                               "to refresh the node.")
        model_path = _resolve_model_path(gguf_model)
        if not os.path.isfile(model_path):
            raise RuntimeError(f"GGUF model not found: {gguf_model} (looked in ComfyUI/models/LLM)")

        map_entries = parse_character_map(character_lora_map)
        style_text = resolve_style(style_medium, style_custom)
        model = get_model(model_path, context_length, gpu_layers)

        seed = int(seed)
        sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "repeat_penalty": repeat_penalty}
        # analysis passes run cooler than the creative writing pass
        analysis = dict(sampling, temperature=min(temperature, 0.5))

        # ---- context budget for the analysis passes
        overhead = count_tokens(model, system_instructions) + 900
        scene_list_budget = max(512, min(4096, num_prompts * 70 + 200))
        story_budget = context_length - overhead - scene_list_budget
        if story_budget < 512:
            raise RuntimeError(
                f"context_length={context_length} is too small for this configuration "
                f"(only {story_budget} tokens left for the story). Increase context_length "
                "or lower num_prompts.")

        working_story = _compress_story(model, story, story_budget, context_length, seed, log)

        # ---- pass 1: character/setting bible
        _check_interrupted()
        log("pass 1/3: building the character/setting bible")
        messages = _bible_messages(working_story, map_entries)
        bible_reply = chat(model, messages, seed=seed + 1,
                           max_tokens=_capped_max_tokens(model, messages, 700, context_length),
                           **analysis)
        bible = merge_bible(map_entries, parse_bible_reply(bible_reply))
        bible_by_name = {e["name"].casefold(): e for e in bible}
        bible_text = format_bible(bible)
        if bible_text:
            log("bible:\n" + bible_text)

        # ---- pass 2: scene selection
        _check_interrupted()
        log(f"pass 2/3: selecting up to {num_prompts} key visual moments")
        known_names = [e["name"] for e in bible]
        messages = _scene_messages(working_story, bible, num_prompts)
        reply = chat(model, messages, seed=seed + 2,
                     max_tokens=_capped_max_tokens(model, messages, scene_list_budget, context_length),
                     **analysis)
        scenes = parse_scene_reply(reply, known_names, num_prompts)
        if not scenes:
            log("scene selection reply was unparseable — retrying once with stricter settings")
            retry = dict(analysis, temperature=0.2)
            reply = chat(model, messages, seed=seed + 3,
                         max_tokens=_capped_max_tokens(model, messages, scene_list_budget,
                                                       context_length),
                         **retry)
            scenes = parse_scene_reply(reply, known_names, num_prompts)
        if not scenes:
            log("scene selection failed twice — falling back to even story chunks")
            chunk_chars = max(400, len(working_story) // max(1, num_prompts) + 200)
            chunks = split_paragraph_chunks(working_story, chunk_chars)[:num_prompts]
            scenes = []
            for chunk in chunks:
                chars = [n for n in known_names
                         if re.search(rf"(?i)\b{re.escape(n)}\b", chunk)]
                scenes.append({"synopsis": re.sub(r"\s+", " ", chunk).strip()[:1500],
                               "characters": chars})
        if len(scenes) < num_prompts:
            log(f"warning: the story supported only {len(scenes)} of the requested "
                f"{num_prompts} scenes — not padding with filler")

        # ---- pass 3: one Krea 2 prompt per scene
        log(f"pass 3/3: writing {len(scenes)} Krea 2 prompts")
        pbar = _make_pbar(len(scenes))
        prompts = []
        for i, scene in enumerate(scenes):
            _check_interrupted()
            messages = _prompt_messages(system_instructions, scene, bible_by_name, style_text)
            raw = chat(model, messages, seed=seed + 100 + i,
                       max_tokens=_capped_max_tokens(model, messages, max_tokens_per_prompt,
                                                     context_length),
                       **sampling)
            triggers = [bible_by_name[n.casefold()]["trigger"] for n in scene["characters"]
                        if bible_by_name.get(n.casefold(), {}).get("trigger")]
            prompt = clean_prompt(raw, triggers)
            if not prompt:
                log(f"warning: scene {i + 1} produced an empty prompt — skipping it")
            else:
                prompts.append(prompt)
                preview = prompt if len(prompt) <= 100 else prompt[:100] + "..."
                log(f"scene {i + 1}/{len(scenes)}: {preview}")
            if pbar is not None:
                pbar.update(1)

        if not prompts:
            raise RuntimeError(
                "The LLM produced no usable prompts. Try an instruct-tuned GGUF model, "
                "a lower temperature, or a larger context_length.")

        # ---- write files + return
        out_dir = _resolve_output_dir(output_dir)
        saved = _write_prompt_files(prompts, out_dir, _sanitize_prefix(filename_prefix))
        log(f"wrote {len(saved)} prompt files to {out_dir}")

        result = {"ui": {"text": prompts},
                  "result": (prompts, len(prompts), bible_text, saved)}
        type(self)._LAST_RESULT = result
        return result


_register_llm_folder()
