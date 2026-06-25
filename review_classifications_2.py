"""
GUI tool to review image classifications.

Keyboard shortcuts:
  Enter          →  Correct
  Backspace/Del  →  Wrong
  N              →  Not Sure
  Left arrow     →  Previous
  Right arrow    →  Next
  R              →  Reset (clears approved, wrong & not-sure CSVs, restarts from scratch)
"""
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import argparse
import csv
import tkinter as tk
from tkinter import font as tkfont, messagebox
from pathlib import Path
from PIL import Image, ImageTk

MAX_IMG_SIZE = (640, 640)   # max display dimensions (w, h)


def load_csv(csv_path):
    """Load (filename, labels) pairs from CSV, skipping header."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                rows.append((row[0].strip(), row[1].strip()))
            elif len(row) == 1:
                rows.append((row[0].strip(), ""))
    return rows


def load_reviewed_set(path):
    """Return set of filenames already written to a review CSV."""
    p = Path(path)
    if not p.exists():
        return set()
    reviewed = set()
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                reviewed.add(row[0].strip())
    return reviewed


class ReviewApp:
    def __init__(self, root, all_entries, approved_csv, wrong_csv, not_sure_csv, image_folder):
        self.root = root
        self.all_entries = all_entries      # full unfiltered list
        self.approved_path = Path(approved_csv)
        self.wrong_path = Path(wrong_csv)
        self.not_sure_path = Path(not_sure_csv)
        self.image_folder = Path(image_folder)
        self.approved_path.parent.mkdir(parents=True, exist_ok=True)

        # Write headers only if files are new
        for p in (self.approved_path, self.wrong_path, self.not_sure_path):
            if not p.exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(["filename", "labels"])

        self._reload_pending()

        # ── Layout ────────────────────────────────────────────────────────────
        root.title("Classification Review")
        root.configure(bg="#212121")
        root.resizable(True, True)

        bold_font = tkfont.Font(family="Helvetica", size=13, weight="bold")
        prog_font = tkfont.Font(family="Helvetica", size=11)

        # Progress
        self.progress_var = tk.StringVar()
        tk.Label(root, textvariable=self.progress_var,
                 font=prog_font, bg="#212121", fg="#aaaaaa").pack(pady=(10, 0))

        # Filename + labels (above image)
        self.info_var = tk.StringVar()
        tk.Label(root, textvariable=self.info_var,
                 font=bold_font, bg="#212121", fg="#ffffff",
                 wraplength=680, justify="center").pack(pady=(6, 4))

        # Image display
        self.img_label = tk.Label(root, bg="#1a1a1a",
                                  width=MAX_IMG_SIZE[0], height=MAX_IMG_SIZE[1])
        self.img_label.pack(padx=16, pady=4)

        # ── Correct / Wrong buttons ───────────────────────────────────────────
        btn_frame = tk.Frame(root, bg="#212121")
        btn_frame.pack(pady=(12, 4))

        self.btn_correct = self._make_button(
            btn_frame, text="✓  Correct",
            bg="#2e7d32", hover_bg="#1b5e20",
            command=self.on_correct, font=bold_font
        )
        self.btn_correct.pack(side="left", padx=16)

        self.btn_wrong = self._make_button(
            btn_frame, text="✗  Wrong",
            bg="#c62828", hover_bg="#7f0000",
            command=self.on_wrong, font=bold_font
        )
        self.btn_wrong.pack(side="left", padx=16)

        self.btn_not_sure = self._make_button(
            btn_frame, text="?  Not Sure",
            bg="#e65100", hover_bg="#bf360c",
            command=self.on_not_sure, font=bold_font
        )
        self.btn_not_sure.pack(side="left", padx=16)

        # ── Previous / Next navigation buttons ───────────────────────────────
        nav_frame = tk.Frame(root, bg="#212121")
        nav_frame.pack(pady=(4, 4))

        self.btn_prev = self._make_button(
            nav_frame, text="◀  Previous",
            bg="#37474f", hover_bg="#263238",
            command=self.on_prev, font=bold_font
        )
        self.btn_prev.pack(side="left", padx=16)

        self.btn_next = self._make_button(
            nav_frame, text="Next  ▶",
            bg="#37474f", hover_bg="#263238",
            command=self.on_next, font=bold_font
        )
        self.btn_next.pack(side="left", padx=16)

        # ── Reset button ──────────────────────────────────────────────────────
        reset_frame = tk.Frame(root, bg="#212121")
        reset_frame.pack(pady=(4, 4))

        self.btn_reset = self._make_button(
            reset_frame, text="↺  Reset",
            bg="#5d4037", hover_bg="#3e2723",
            command=self.on_reset, font=bold_font
        )
        self.btn_reset.pack()

        # Hint label
        tk.Label(root,
                 text="Enter Correct   BackSpace Wrong   N Not Sure   ← Previous   → Next   R Reset",
                 font=tkfont.Font(family="Helvetica", size=9),
                 bg="#212121", fg="#666666").pack(pady=(0, 8))

        # ── Keyboard shortcuts ────────────────────────────────────────────────
        root.bind("<Return>",    lambda _: self.on_correct())
        root.bind("<BackSpace>", lambda _: self.on_wrong())
        root.bind("<Delete>",    lambda _: self.on_wrong())
        root.bind("<Left>",      lambda _: self.on_prev())
        root.bind("<Right>",     lambda _: self.on_next())
        root.bind("<n>",         lambda _: self.on_not_sure())
        root.bind("<N>",         lambda _: self.on_not_sure())
        root.bind("<r>",         lambda _: self.on_reset())
        root.bind("<R>",         lambda _: self.on_reset())

        self._show_current()

    # ── Internal state helpers ────────────────────────────────────────────────

    def _reload_pending(self):
        """Recompute the pending list from all_entries minus already-reviewed."""
        approved  = load_reviewed_set(self.approved_path)
        wrong     = load_reviewed_set(self.wrong_path)
        not_sure  = load_reviewed_set(self.not_sure_path)
        done = approved | wrong | not_sure
        self.entries = [(f, l) for f, l in self.all_entries if f not in done]
        self.index = 0
        self.approved_this_session: set  = set()
        self.wrong_this_session: set     = set()
        self.not_sure_this_session: set  = set()

    # ── Button helpers ────────────────────────────────────────────────────────

    def _make_button(self, parent, text, bg, hover_bg, command, font):
        """Create a colored label-button that works on macOS."""
        lbl = tk.Label(
            parent, text=text, bg=bg, fg="white", font=font,
            padx=24, pady=10, cursor="hand2", relief="flat"
        )
        lbl.bind("<Enter>",         lambda _, w=lbl, c=hover_bg: w.config(bg=c))
        lbl.bind("<Leave>",         lambda _, w=lbl, c=bg:       w.config(bg=c))
        lbl.bind("<ButtonPress-1>", lambda _, fn=command:        fn())
        lbl._default_bg = bg
        lbl._hover_bg   = hover_bg
        return lbl

    def _disable_judgment_buttons(self):
        for btn in (self.btn_correct, self.btn_wrong, self.btn_not_sure):
            btn.config(bg="#555555", cursor="")
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
            btn.unbind("<ButtonPress-1>")

    def _enable_judgment_buttons(self):
        for btn, fn in ((self.btn_correct,  self.on_correct),
                        (self.btn_wrong,    self.on_wrong),
                        (self.btn_not_sure, self.on_not_sure)):
            bg       = btn._default_bg
            hover_bg = btn._hover_bg
            btn.config(bg=bg, cursor="hand2")
            btn.bind("<Enter>",         lambda _, w=btn, c=hover_bg: w.config(bg=c))
            btn.bind("<Leave>",         lambda _, w=btn, c=bg:       w.config(bg=c))
            btn.bind("<ButtonPress-1>", lambda _, f=fn:              f())

    def _update_nav_buttons(self):
        """Grey out Prev when at start, Next when at end."""
        if self.index <= 0:
            self.btn_prev.config(bg="#555555", cursor="")
        else:
            self.btn_prev.config(bg="#37474f", cursor="hand2")
        if self.index >= len(self.entries) - 1:
            self.btn_next.config(bg="#555555", cursor="")
        else:
            self.btn_next.config(bg="#37474f", cursor="hand2")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_current(self):
        if self.index >= len(self.entries):
            total = len(self.all_entries)
            done  = total - len(self.entries) + len(self.approved_this_session) + len(self.wrong_this_session)
            self.info_var.set("All pending images reviewed!")
            self.progress_var.set(f"{done} / {total} total reviewed")
            self.img_label.config(image="", text="Done", fg="#aaaaaa",
                                  font=tkfont.Font(size=24), width=20, height=4)
            self._disable_judgment_buttons()
            self._update_nav_buttons()
            return

        self._update_nav_buttons()
        self._enable_judgment_buttons()
        filename, labels = self.entries[self.index]

        if filename in self.approved_this_session:
            mark = "  ✓"
        elif filename in self.wrong_this_session:
            mark = "  ✗"
        elif filename in self.not_sure_this_session:
            mark = "  ?"
        else:
            mark = ""

        self.progress_var.set(f"{self.index + 1} / {len(self.entries)}{mark}")
        label_display = labels if labels else "(no label)"
        self.info_var.set(f"{filename}\n{label_display}")

        img_path = self.image_folder / filename
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            img.thumbnail(MAX_IMG_SIZE, Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self._tk_img, text="",
                                  width=img.width, height=img.height)
        else:
            self._tk_img = None
            self.img_label.config(image="",
                                  text=f"[Image not found:\n{filename}]",
                                  fg="#ff5555",
                                  font=tkfont.Font(size=12),
                                  width=40, height=6)

    def on_correct(self):
        if self.index >= len(self.entries):
            return
        filename, labels = self.entries[self.index]
        if filename not in self.approved_this_session:
            with open(self.approved_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([filename, labels])
            print(f"[OK]  {filename!s:<50} {labels}")
        self.approved_this_session.add(filename)
        self.wrong_this_session.discard(filename)
        self.not_sure_this_session.discard(filename)
        self.index += 1
        self._show_current()

    def on_wrong(self):
        if self.index >= len(self.entries):
            return
        filename, labels = self.entries[self.index]
        if filename not in self.wrong_this_session:
            with open(self.wrong_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([filename, labels])
            print(f"[X]   {filename!s:<50} {labels}")
        self.wrong_this_session.add(filename)
        self.approved_this_session.discard(filename)
        self.not_sure_this_session.discard(filename)
        self.index += 1
        self._show_current()

    def on_not_sure(self):
        if self.index >= len(self.entries):
            return
        filename, labels = self.entries[self.index]
        if filename not in self.not_sure_this_session:
            with open(self.not_sure_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([filename, labels])
            print(f"[?]   {filename!s:<50} {labels}")
        self.not_sure_this_session.add(filename)
        self.approved_this_session.discard(filename)
        self.wrong_this_session.discard(filename)
        self.index += 1
        self._show_current()

    def on_prev(self):
        if self.index > 0:
            self.index -= 1
            self._show_current()

    def on_next(self):
        if self.index < len(self.entries) - 1:
            self.index += 1
            self._show_current()

    def on_reset(self):
        from tkinter import messagebox
        if not messagebox.askyesno(
            "Reset",
            "This will delete all approved, wrong and not-sure records for this band\n"
            "and restart the review from scratch.\n\nContinue?",
            icon="warning"
        ):
            return
        for p in (self.approved_path, self.wrong_path, self.not_sure_path):
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["filename", "labels"])
        print("[RESET] Cleared approved, wrong and not-sure CSVs.")
        self._reload_pending()
        self._enable_judgment_buttons()
        self._show_current()


def main():
    parser = argparse.ArgumentParser(description="Review image classifications.")
    parser.add_argument("--panel", default="23-P09-C", help="Panel identifier, e.g. 23-P09-C")
    parser.add_argument("--mode", default="EL", help="Image band to review (e.g. EL, VI, VIT)")
    args = parser.parse_args()

    panel = args.panel
    mode = args.mode
    image_folder = f"normalized_images/{panel}/{mode}"
    labels_csv   = f"OPENAI/{panel}/classification_results_{mode}_{panel}.csv"
    approved_csv = f"OPENAI/{panel}/approved_classifications_{mode}_{panel}.csv"
    wrong_csv    = f"OPENAI/{panel}/wrong_classifications_{mode}_{panel}.csv"
    not_sure_csv = f"OPENAI/{panel}/not_sure_classifications_{mode}_{panel}.csv"

    labels_path = Path(labels_csv)
    if not labels_path.exists():
        print(f"ERROR: Labels CSV not found: {labels_csv}")
        return

    all_entries = sorted(load_csv(labels_path), key=lambda x: (x[1].strip().lower() == 'good', x[0]))
    if not all_entries:
        print(f"No entries found in {labels_csv}")
        return

    approved = load_reviewed_set(approved_csv)
    wrong    = load_reviewed_set(wrong_csv)
    not_sure = load_reviewed_set(not_sure_csv)
    pending  = len(all_entries) - len(approved | wrong | not_sure)

    print(f"Total: {len(all_entries)}  |  approved: {len(approved)}  |  "
          f"wrong: {len(wrong)}  |  not sure: {len(not_sure)}  |  pending: {pending}")
    print(f"Approved  → {approved_csv}")
    print(f"Wrong     → {wrong_csv}")
    print(f"Not sure  → {not_sure_csv}\n")

    if pending == 0:
        print("All images reviewed. Launch anyway to use Reset.")

    root = tk.Tk()
    ReviewApp(root, all_entries, approved_csv, wrong_csv, not_sure_csv, image_folder)
    # Bring window to front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()


if __name__ == "__main__":
    main()
