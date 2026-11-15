"""Team possession from automatic tracking: nearest player, gap filling, majority-vote smoothing.

One command regenerates the submission file from the provided tracking data:

    python possession/run.py

It writes possession/output/M2EN.csv (frame,label), 199319 rows. The team name is set in TEAM
below and must match the registered name exactly; --team overrides it.

Method, in three steps
----------------------
1. Nearest player. For every frame where the tracking sees the ball, the team of the tracked
   player closest to the ball is given the ball. Frames without a ball stay undecided.
2. Gap filling. When the ball is not tracked for at most MAX_GAP frames, the last known team keeps
   the ball. Longer gaps stay undecided and become D (dead ball) in step 3.
3. Smoothing. A centred majority vote over +/- WINDOW frames removes one-frame flicker.

Parameters are fixed by football reasoning, not tuned on the ground truth or on the video
(the competition rules forbid that):

  MAX_GAP = 125 frames (5 s at 25 fps). A long goalkeeper clearance keeps the ball in the air for
            roughly 2 to 4 seconds, often out of sight of the tracking, and possession does not
            change during that time. A gap beyond 5 s looks more like a stoppage (throw-in, free
            kick, treatment) than an occlusion.
  WINDOW  = 50 frames (+/- 2 s). Long enough to remove flicker, short enough not to erase real
            possessions of a few seconds.

Known limits
------------
* If the ball lands with the opponent while it is untracked, the change is only seen when the ball
  is tracked again.
* The vote can pull the edges of a long gap towards a team by up to WINDOW frames.
* Only team of the nearest player is used: no notion of control, no Smart Tagging, no homography.

Only the tactical tracking (BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt) is read.
numpy only; it must be run from a clone of the challenge repository because it imports games.py.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
TEAM = "M2EN"  # team name as registered: the submission file is <TEAM>.csv
OUT_DIR = os.path.join(HERE, "output")
MAX_GAP = 125  # frames: 5 s at 25 fps, see the module docstring
WINDOW = 50    # half-width of the majority-vote window, in frames (+/- 2 s)

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed" with NaN for a
# missing observation. Splitting on ':' gives exactly three parts.
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN", which is all we need to test for

HOME, AWAY, NOBODY = 0, 1, -1
LABELS = np.array(["H", "A", "D"])


def team_of_each_track(meta_path):
    """Array over track slots: 0 if the slot belongs to the home team, 1 if away.

    The first team in the ATD metadata is the home side (FIFA/EPTS convention). A slot is a
    tracklet, not a person: there are 46 of them for 22 players on the pitch.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names, for printing."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Per-frame team of the player closest to the ball. NOBODY where there is no answer.

    Coordinates are normalised 0-1 and the pitch is not square, so both axes are scaled to metres
    before distances are compared.
    """
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == MISSING:
                continue  # ball not tracked: left undecided, handled by fill_gaps

            comma = ball.index(",")
            ball_x = float(ball[:comma]) * length
            ball_y = float(ball[comma + 1:ball.index(",", comma + 1)]) * width

            best, best_team = None, NOBODY
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                dx = float(group[:comma]) * length - ball_x
                dy = float(group[comma + 1:group.index(",", comma + 1)]) * width - ball_y
                distance = dx * dx + dy * dy
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]

            out[int(parts[0]) - 1] = best_team  # frame_id counts from 1, submissions from 0
    return out


def fill_gaps(codes, max_gap):
    """Carry the last known team across short stretches where the ball is not tracked.

    A gap of at most max_gap frames, preceded by a known team, takes that team. Longer gaps, and
    gaps before the first tracked ball, stay NOBODY (they become D after smoothing).
    """
    out = codes.copy()
    n = len(codes)
    last = NOBODY
    i = 0
    while i < n:
        if codes[i] != NOBODY:
            last = codes[i]
            i += 1
            continue
        j = i
        while j < n and codes[j] == NOBODY:
            j += 1
        if last != NOBODY and (j - i) <= max_gap:
            out[i:j] = last
        i = j
    return out


def smooth(codes, window):
    """Majority vote over a centred window of +/- `window` frames. D where nobody has the ball.

    Computed with cumulative sums, so the cost does not grow with the window. Ties go to home.
    """
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cumulative[high] - home_cumulative[low]
    away = away_cumulative[high] - away_cumulative[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def write_submission(path, labels):
    """A frame,label CSV, one row per frame, frame counted from 0."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def build(game=GAME, data_dir=None):
    """Per-frame team codes for the whole video, straight from the tracking."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes = nearest_player_labels(raw_path, team_of_each_track(meta_path), n_frames,
                                  GAMES[game]["pitch"])
    seen = int(np.sum(codes != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)".format(
        seen, n_frames, 100.0 * seen / n_frames))
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default=TEAM,
                        help="team name exactly as registered; the file is <team>.csv")
    parser.add_argument("--max-gap", type=int, default=MAX_GAP,
                        help="longest untracked-ball gap (frames) that keeps the last team")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help="half-width of the majority-vote window, in frames")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    args = parser.parse_args()

    codes = build()
    tracked = int(np.sum(codes != NOBODY))
    codes = fill_gaps(codes, args.max_gap)
    filled = int(np.sum(codes != NOBODY)) - tracked
    print("  gap filling (<= {} frames): {} frames given the last known team".format(
        args.max_gap, filled))

    labels = smooth(codes, args.window)
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.team)), labels)

    counts = {label: int(np.sum(labels == label)) for label in LABELS}
    print("\nH {:>6}  A {:>6}  D {:>6}".format(counts["H"], counts["A"], counts["D"]))
    print("-> {}".format(os.path.relpath(path, ROOT)))
    print("\nNow check the format:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()