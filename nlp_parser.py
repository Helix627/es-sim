"""
Open-Vocabulary Natural Language Command Parser for Vision-Guided Robotic Arm.
Supports:
  1. Manual two-object naming (`create_task(pick_name, place_name)`):
     e.g., Pick = "red bottle", Place = "white breadboard"
  2. Two-target pick-and-place sentences:
     e.g., "Pick up the red bottle and place it in the blue tray."
     e.g., "Grab the white mixed foam board and put it on the cardboard box."
     e.g., "yellow cup -> white breadboard" or "black phone, blue tray"
  3. Single-target locate/pick queries:
     e.g., "where is the white mixed foam board"
"""
import re
from dataclasses import dataclass
from typing import Optional

KNOWN_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "cyan", "white", "black", "grey", "gray", "brown", "silver", "gold"
]


@dataclass
class PickAndPlaceTask:
    raw_command: str
    pick_phrase: str
    place_phrase: Optional[str]
    pick_color: str
    pick_item: str
    place_color: str
    place_item: str
    is_two_object_task: bool = True
    valid: bool = True
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if not self.valid:
            return f"Invalid Command: {self.error_message}"
        if self.place_phrase:
            return f"Task: Pick [{self.pick_phrase}] --> Place at [{self.place_phrase}]"
        return f"Task: Locate & Pick [{self.pick_phrase}] --> Place at [Default Drop Zone]"


def _clean_phrase(text: str) -> str:
    """Strips leading articles, polite words, and trailing punctuation."""
    t = text.strip().strip(".?!,;:\"'")
    t = re.sub(
        r"^(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:where\s+is\s+|find\s+|locate\s+|pick\s+up\s+|pick\s+|grab\s+|take\s+|get\s+|lift\s+|move\s+|point\s+to\s+|place\s+in\s+|place\s+on\s+|put\s+in\s+|put\s+on\s+)?",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    t = re.sub(r"^(?:the|a|an|my|that|this)\s+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+(?:on\s+the\s+desk|on\s+the\s+table|in\s+the\s+image|from\s+the\s+table)$", "", t, flags=re.IGNORECASE).strip()
    return t.lower()


def _extract_color_and_item(phrase: str, default_item: str = "object") -> tuple[str, str]:
    words = re.findall(r"[a-zA-Z0-9#]+", phrase.lower())
    color = ""
    for w in words:
        if w in KNOWN_COLORS:
            color = "grey" if w == "gray" else w
            break
    item_words = [w for w in words if w != color and w not in ("color", "coloured", "colored")]
    item = " ".join(item_words) if item_words else default_item
    return color, item


class CommandParser:
    def __init__(self):
        # Matches: "<pick verb> <source phrase> and <place verb> [it] in/on/to <dest phrase>"
        self.two_part_pattern = re.compile(
            r"^(?:please\s+)?(?:pick\s*up|pick|grab|take|get|lift|move|transfer)?\s*"
            r"(?P<src>.+?)\s+"
            r"(?:and\s+|,\s*then\s+|,\s*)?"
            r"(?:(?:place|put|drop|move|deposit|set)\s+(?:it\s+|them\s+)?)?"
            r"(?:in|into|on|onto|at|to|inside|->|=>)\s+"
            r"(?P<dst>.+?)[.?!]*$",
            re.IGNORECASE,
        )
        # Matches shorthand: "<pick object> , <place object>" or "<pick object> | <place object>"
        self.delim_pattern = re.compile(
            r"^(?P<src>[^,|;]+?)\s*(?:->|=>|\||;|,)\s*(?P<dst>[^,|;]+?)$",
            re.IGNORECASE,
        )

    def create_task(self, pick_name: str, place_name: Optional[str] = None) -> PickAndPlaceTask:
        """Directly constructs a PickAndPlaceTask from manual pick & place object names."""
        src_phrase = _clean_phrase(pick_name) or pick_name.strip().lower()
        dst_phrase = (_clean_phrase(place_name) or place_name.strip().lower()) if (place_name and place_name.strip()) else None

        p_col, p_item = _extract_color_and_item(src_phrase, "object")
        if dst_phrase:
            d_col, d_item = _extract_color_and_item(dst_phrase, "tray")
            raw = f"Pick up [{src_phrase}] -> Place on [{dst_phrase}]"
            return PickAndPlaceTask(
                raw_command=raw,
                pick_phrase=src_phrase,
                place_phrase=dst_phrase,
                pick_color=p_col or "target",
                pick_item=p_item,
                place_color=d_col or "dest",
                place_item=d_item,
                is_two_object_task=True,
                valid=bool(src_phrase),
            )
        else:
            raw = f"Pick up [{src_phrase}]"
            return PickAndPlaceTask(
                raw_command=raw,
                pick_phrase=src_phrase,
                place_phrase=None,
                pick_color=p_col or "target",
                pick_item=p_item,
                place_color="blue",
                place_item="drop zone",
                is_two_object_task=False,
                valid=bool(src_phrase),
            )

    def parse(self, text: str) -> PickAndPlaceTask:
        cleaned = text.strip()
        if not cleaned:
            return PickAndPlaceTask(
                raw_command=text,
                pick_phrase="",
                place_phrase=None,
                pick_color="",
                pick_item="",
                place_color="",
                place_item="",
                is_two_object_task=False,
                valid=False,
                error_message="Empty command.",
            )

        # 1. Check for shorthand delimiter ("red bottle -> white breadboard" or "red bottle, blue tray")
        m_delim = self.delim_pattern.match(cleaned)
        if m_delim and "pick" not in cleaned.lower() and "place" not in cleaned.lower():
            src_phrase = _clean_phrase(m_delim.group("src"))
            dst_phrase = _clean_phrase(m_delim.group("dst"))
            if src_phrase and dst_phrase:
                return self.create_task(src_phrase, dst_phrase)

        # 2. Check for natural language two-object Pick & Place command
        m = self.two_part_pattern.match(cleaned)
        if m:
            src_phrase = _clean_phrase(m.group("src"))
            dst_phrase = _clean_phrase(m.group("dst"))
            if src_phrase and dst_phrase:
                p_col, p_item = _extract_color_and_item(src_phrase, "object")
                d_col, d_item = _extract_color_and_item(dst_phrase, "tray")
                return PickAndPlaceTask(
                    raw_command=cleaned,
                    pick_phrase=src_phrase,
                    place_phrase=dst_phrase,
                    pick_color=p_col or "target",
                    pick_item=p_item,
                    place_color=d_col or "dest",
                    place_item=d_item,
                    is_two_object_task=True,
                    valid=True,
                )

        # 3. Single-object locate / pick command
        src_phrase = _clean_phrase(cleaned)
        if not src_phrase:
            src_phrase = cleaned.lower()
        p_col, p_item = _extract_color_and_item(src_phrase, "object")
        return PickAndPlaceTask(
            raw_command=cleaned,
            pick_phrase=src_phrase,
            place_phrase=None,
            pick_color=p_col or "target",
            pick_item=p_item,
            place_color="blue",
            place_item="drop zone",
            is_two_object_task=False,
            valid=True,
        )
