import os
import logging
import numpy as np
import re
import ffmpeg
import srt


def check_exists(output, force):
    if os.path.exists(output):
        if force:
            logging.info(f"{output} exists. Will overwrite it")
        else:
            logging.info(
                f"{output} exists, skipping... Use the --force flag to overwrite"
            )
            return True
    return False

def load_audio(file: str, sr: int = 16000) -> np.ndarray:
    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def remove_short_segments(segments, threshold):
     #Remove segments whose length < threshold
    return [s for s in segments if s["end"] - s["start"] > threshold]

def expand_segments(segments, expand_head, expand_tail, total_length):
    # Pad head and tail for each time segment
    results = []
    for i in range(len(segments)):
        t = segments[i]
        start = max(t["start"] - expand_head, segments[i - 1]["end"] if i > 0 else 0)
        end = min(
            t["end"] + expand_tail,
            segments[i + 1]["start"] if i < len(segments) - 1 else total_length,
        )
        results.append({"start": start, "end": end})
    return results

def merge_adjacent_segments(segments, threshold):
    # Merge two adjacent segments if their distance < threshold
    results = []
    i = 0
    while i < len(segments):
        s = segments[i]
        for j in range(i + 1, len(segments)):
            if segments[j]["start"] < s["end"] + threshold:
                s["end"] = segments[j]["end"]
                i = j
            else:
                break
        i += 1
        results.append(s)
    return results

# a very simple markdown parser
class MD:
    def __init__(self, filename, encoding):
        self.lines = []
        self.EDIT_DONE_MAKR = "<-- Mark if you are done editing."
        self.encoding = encoding
        self.filename = filename
        if filename:
            self.load_file()

    def load_file(self):
        if os.path.exists(self.filename):
            with open(self.filename, encoding=self.encoding) as f:
                self.lines = f.readlines()

    def clear(self):
        self.lines = []

    def write(self):
        with open(self.filename, "wb") as f:
            f.write("\n".join(self.lines).encode(self.encoding, "replace"))

    def tasks(self):
        # get all tasks with their status
        ret = []
        for l in self.lines:
            mark, task = self._parse_task_status(l)
            if mark is not None:
                ret.append((mark, task))
        return ret

    def done_editing(self):
        for m, t in self.tasks():
            if m and self.EDIT_DONE_MAKR in t:
                return True
        return False

    def add(self, line):
        self.lines.append(line)

    def add_task(self, mark, contents):
        self.add(f'- [{"x" if mark else " "}] {contents.strip()}')

    def add_done_editing(self, mark):
        self.add_task(mark, self.EDIT_DONE_MAKR)

    def add_video(self, video_fn):
        ext = os.path.splitext(video_fn)[1][1:]
        self.add(
            f'\n<video controls="true" allowfullscreen="true"> <source src="{video_fn}" type="video/{ext}"> </video>\n'
        )

    def _parse_task_status(self, line):
        # return (is_marked, rest) or (None, line) if not a task
        m = re.match(r"- +\[([ x])\] +(.*)", line)
        if not m:
            return None, line
        return m.groups()[0].lower() == "x", m.groups()[1]