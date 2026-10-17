"""End takes that turn into a drone, before the drone is heard.

A take can derail into a low, steady drone that never ends: the model cycles through
a few dozen quiet codes that are not in the pause fix's silence codes, so neither the
pause fix nor the server's silence guard catches it, and it runs to the length cap.
The idea is T8star's ComfyUI-Drama3-T8 (few distinct codes in a recent window); the
windows and limits here were fitted on 96 takes of the full model (12 seeds x 4 lines x
top-k 30 / all): normal speech has at least 40 distinct main codes in any 64 frames,
the three derailed takes fell to 22-23.

- Hold: while the last SHORT_WINDOW frames have at most SHORT_MAX distinct main codes,
  the frame loop stops reporting new frames, so they are not vocoded or streamed yet.
  In normal speech that happens on 1 % of frames, for at most 18 frames (0.8 s).
- Stall: when the last LONG_WINDOW frames have at most LONG_MAX distinct codes, the
  take ends as "stall" where the hold began; the held frames are dropped. A take that
  ends any other way first releases what it held.
"""

from __future__ import annotations

import collections
import os

ENABLED = os.environ.get("DRAMA3_STALL_GUARD", "1") != "0"
SHORT_WINDOW, SHORT_MAX = 24, 18
LONG_WINDOW, LONG_MAX = 64, 30


class StallGuard:
    def __init__(self) -> None:
        self.recent: collections.deque[int] = collections.deque(maxlen=LONG_WINDOW)
        self.released = 0  # frames reported to the streaming loop

    def add(self, main: int, produced: int) -> tuple[bool, range]:
        """Record frame `produced` (its main code); returns (stalled, frame counts to
        report now). Without a stall, the counts run on from the last one reported."""
        if not ENABLED:
            return False, range(produced, produced + 1)
        self.recent.append(main)
        if len(self.recent) == LONG_WINDOW and len(set(self.recent)) <= LONG_MAX:
            return True, range(0)
        short = list(self.recent)[-SHORT_WINDOW:]
        if (len(short) == SHORT_WINDOW and len(set(short)) <= SHORT_MAX
                and produced - self.released < LONG_WINDOW):
            return False, range(0)  # hold
        return False, self.release(produced)

    def release(self, produced: int) -> range:
        """All frame counts not reported yet, up to produced (before a normal end)."""
        counts = range(self.released + 1, produced + 1)
        self.released = max(self.released, produced)
        return counts
