# AI Study Planner

A Python mini project that turns a raw syllabus into a day-by-day study schedule.

You paste your syllabus text, set an exam date and the hours you can study per day. An LLM (Google Gemini) breaks the syllabus into topics with estimated hours and difficulty. Plain Python then builds the schedule, hardest topics first, with revision days kept free before the exam.

## Python concepts used

- Classes and objects (dataclasses)
- Lists and dictionaries
- Functions with type hints
- LLM / API calling (Google Gen AI SDK)
- JSON parsing and validation

## How it works

1. **Parse**: the syllabus text is sent to Gemini, which returns JSON: a list of topics with `name`, `hours` and `difficulty` (1-5).
2. **Validate**: the JSON is converted into `Topic` objects, and difficulty is clamped to the range 1-5.
3. **Schedule**: topics are sorted by difficulty (then by hours), and each day is filled up to the daily hour limit. A big topic can be split across several days.
4. **Revise**: the last few days before the exam are reserved for revision.
5. **Warn**: if the syllabus does not fit in the available time, the leftover topics are reported instead of being silently dropped.

## Setup

Requires Python 3.10 or newer.

```bash
pip install -U google-genai
```

Set your Gemini API key as an environment variable (never paste it into code you push to GitHub):

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
```

In Google Colab, store the key in **Secrets** and load it with `userdata.get("GEMINI_API_KEY")`.

## The notebook, cell by cell

Run the cells in order from top to bottom. If you restart the kernel, run Cells 2, 3 and 4 again before Cell 5.

### Cell 1: Install

```python
!pip install -U google-genai
```

### Cell 2: Data models

```python
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class Topic:
    name: str
    hours: float
    difficulty: int  # 1 to 5
    status: str = "pending"


@dataclass
class Syllabus:
    subject: str
    topics: list[Topic] = field(default_factory=list)

    def add_topic(self, topic: Topic) -> None:
        self.topics.append(topic)

    def pending_topics(self) -> list[Topic]:
        return [t for t in self.topics if t.status == "pending"]

    def total_hours(self) -> float:
        return sum(t.hours for t in self.pending_topics())


@dataclass
class StudyDay:
    day: date
    items: list[tuple[str, float]] = field(default_factory=list)  # (topic name, hours)
    is_revision: bool = False

    @property
    def hours_used(self) -> float:
        return sum(hours for _, hours in self.items)

    def add(self, topic_name: str, hours: float) -> None:
        self.items.append((topic_name, hours))
```

### Cell 3: LLM parsing (Google Gemini)

```python
import json
from google import genai
from google.genai import types

# Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment
client = genai.Client()

SYSTEM_PROMPT = """You are a study planning assistant.
Given a syllabus, break it into study topics.
Return ONLY valid JSON in this shape:
{"topics": [{"name": "string", "hours": number, "difficulty": integer 1-5}]}
Rules:
- hours = realistic study hours for an average student
- difficulty: 1 = easy, 5 = very hard
- keep topic names short"""


def call_llm(system: str, user: str) -> str:
    """The only function that talks to the provider. Swap this to change models."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        ),
    )
    return response.text


def clean_json(raw: str) -> str:
    """Remove ```json fences if the model adds them anyway."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def parse_syllabus(text: str) -> list[Topic]:
    raw = call_llm(SYSTEM_PROMPT, f"Syllabus:\n{text}")
    data: dict = json.loads(clean_json(raw))
    return [
        Topic(
            name=str(item["name"]),
            hours=float(item["hours"]),
            difficulty=max(1, min(5, int(item["difficulty"]))),  # clamp to 1-5
        )
        for item in data["topics"]
    ]
```

### Cell 4: Scheduler

```python
class Planner:
    def __init__(
        self,
        syllabus: Syllabus,
        exam_date: date,
        daily_hours: float,
        revision_days: int = 2,
        start: date | None = None,
    ) -> None:
        self.syllabus = syllabus
        self.exam_date = exam_date
        self.daily_hours = daily_hours
        self.revision_days = revision_days
        self.start = start or date.today()
        self.unscheduled: dict[str, float] = {}  # topics that did not fit

    def _study_dates(self) -> list[date]:
        """Days from start until the revision window begins."""
        last_study_day = self.exam_date - timedelta(days=self.revision_days + 1)
        dates: list[date] = []
        current = self.start
        while current <= last_study_day:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def _revision_dates(self) -> list[date]:
        return [
            self.exam_date - timedelta(days=n)
            for n in range(self.revision_days, 0, -1)
        ]

    def build_schedule(self) -> list[StudyDay]:
        study_dates = self._study_dates()
        if not study_dates:
            raise ValueError("Not enough days before the exam. Check your dates.")

        schedule = [StudyDay(d) for d in study_dates]

        # hardest first; longer topics first when difficulty is equal
        topics = sorted(
            self.syllabus.pending_topics(),
            key=lambda t: (-t.difficulty, -t.hours),
        )

        self.unscheduled = {}
        i = 0  # index of the day currently being filled

        for topic in topics:
            remaining = topic.hours
            while remaining > 1e-9:
                if i >= len(schedule):  # no days left
                    self.unscheduled[topic.name] = remaining
                    break

                free = self.daily_hours - schedule[i].hours_used
                if free <= 1e-9:
                    i += 1
                    continue

                chunk = min(free, remaining)
                schedule[i].add(topic.name, round(chunk, 2))
                remaining -= chunk

        # revision days at the end
        for d in self._revision_dates():
            rev = StudyDay(d, is_revision=True)
            rev.add("Revision", self.daily_hours)
            schedule.append(rev)

        return schedule

    def print_schedule(self, schedule: list[StudyDay]) -> None:
        for sd in schedule:
            print(f"\n{sd.day.strftime('%a, %d %b')}  ({sd.hours_used:g}h)")
            for name, hours in sd.items:
                print(f"   - {name}: {hours:g}h")

        if self.unscheduled:
            print("\nWARNING: not enough time for these topics:")
            for name, hours in self.unscheduled.items():
                print(f"   - {name}: {hours:g}h left over")
            print("Increase daily hours or move the exam date.")
```

### Cell 5: Run everything

```python
syllabus_text = """
Unit 1: Python basics, lists, dictionaries
Unit 2: Functions, args and kwargs, decorators
Unit 3: OOP, classes, inheritance
Unit 4: File handling and APIs
"""

# LLM turns the text into Topic objects
syllabus = Syllabus(subject="Python")
for topic in parse_syllabus(syllabus_text):
    syllabus.add_topic(topic)

print("Topics found:")
for t in syllabus.topics:
    print(f"  {t.name} | {t.hours}h | difficulty {t.difficulty}")

# Plain Python builds the schedule
planner = Planner(
    syllabus,
    exam_date=date.today() + timedelta(days=10),
    daily_hours=3,
)
schedule = planner.build_schedule()
planner.print_schedule(schedule)
```

## Example output

```
Topics found:
  Python Basics | 5.0h | difficulty 2
  Lists | 3.0h | difficulty 2
  Dictionaries | 3.0h | difficulty 2
  Functions | 4.0h | difficulty 2
  Args and Kwargs | 2.0h | difficulty 3
  Decorators | 5.0h | difficulty 4
  OOP Concepts | 4.0h | difficulty 3
  Classes & Objects | 5.0h | difficulty 3
  Inheritance | 4.0h | difficulty 3
  File Handling | 4.0h | difficulty 3
  Working with APIs | 6.0h | difficulty 4

Wed, 07 Oct  (3h)
   - Working with APIs: 3h

Thu, 08 Oct  (3h)
   - Working with APIs: 3h

Fri, 09 Oct  (3h)
   - Decorators: 3h
...
```

If the total topic hours are more than the available study time, the planner prints a warning listing the topics (and hours) that did not fit. To fix that, increase `daily_hours` or move the exam date further away.

## Known limitations

- Topics are scheduled hardest first, so an easy foundational topic (such as Python Basics) can end up last or be the one that does not fit. Prerequisite ordering is not handled yet.
- The LLM can occasionally return invalid JSON, and the program currently stops with an error in that case.

## Roadmap

- [ ] `@timer` decorator to measure LLM call time
- [ ] `@retry` decorator to retry when the LLM returns bad JSON
- [ ] `@log_call` decorator to log actions to a file
- [ ] `mark_done()` and `replan()` to adjust the plan as topics are completed
- [ ] Prerequisite-aware ordering of topics
- [ ] Export the schedule to JSON / CSV
- [ ] Simple Streamlit interface

## Tech stack

Python 3.10+, Google Gen AI SDK (`google-genai`), Jupyter / Google Colab
