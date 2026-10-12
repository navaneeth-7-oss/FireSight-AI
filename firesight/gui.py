from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import shutil
from tkinter import BooleanVar, StringVar, filedialog, messagebox, ttk
import tkinter as tk
from urllib.parse import urlsplit

import cv2
from PIL import Image, ImageTk

from .config import AppConfig, build_rtsp_url, save_config
from .engine import FireSightEngine, FrameResult, run_video_source, test_video_connection
from .events import sanitize_source


BG = "#0b1118"
SIDEBAR = "#0d151d"
PANEL = "#121d27"
PANEL2 = "#192733"
TEXT = "#e5edf3"
MUTED = "#8b9ba9"
ACCENT = "#34b7b0"
GREEN = "#38c59a"
AMBER = "#e8b255"
CRITICAL = "#d94f55"


class FireSightDashboard:
    """Responsive local CCTV monitoring dashboard with background inference."""

    def __init__(self, root: tk.Tk, config: AppConfig) -> None:
        self.root, self.config = root, config
        self.engine = FireSightEngine(config)
        self.messages: queue.Queue[tuple] = queue.Queue()
        self.frames: queue.Queue[FrameResult] = queue.Queue(maxsize=1)
        self.stop_event, self.busy = threading.Event(), threading.Event()
        self._closed = self._shutdown_pending = False
        self._worker_thread: threading.Thread | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._current_events: dict[str, dict] = {}
        self._active_event_id: str | None = None
        self._active_event_summary: str | None = None
        self._alert_acknowledged = False
        self._silenced = False
        self._last_alert_state = "NORMAL"
        self._frame_times: list[float] = []
        self._frames_processed = 0
        self._reconnect_attempts = 0
        self._last_frame_monotonic: float | None = None
        self._resolution = "—"
        self._counts = {"fire": 0, "smoke": 0, "person": 0}
        self._connected = False
        self._rtsp_source: str | None = None
        self._current_source: tuple[str | int, str, bool] | None = None
        self._pending_action = None
        self._camera_name = config.camera_name or "Laptop Webcam"
        self._source_type = "WEBCAM"
        self._last_detection_detail = "No detections in the latest frame"
        self.root.title("FireSight | Security Operations")
        self.root.geometry("1500x940")
        self.root.minsize(1050, 700)
        self.root.configure(bg=BG)
        self._style()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._poll)
        self._status("Loading model…", AMBER)
        threading.Thread(target=self._load_model, name="firesight-model-loader", daemon=True).start()

    def _style(self) -> None:
        s = ttk.Style(self.root)
        try: s.theme_use("clam")
        except tk.TclError: pass
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Muted.Panel.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        s.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 19, "bold"))
        s.configure("CardValue.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 17, "bold"))
        s.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(10, 7), background=PANEL2, foreground=TEXT, relief="flat", borderwidth=0, focusthickness=0)
        s.map("TButton", background=[("active", "#233945"), ("disabled", "#14202a")], foreground=[("disabled", "#647582")], relief=[("pressed", "flat"), ("!pressed", "flat")])
        s.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), padding=(12, 8), background="#1c7775", foreground="#f1ffff", relief="flat", borderwidth=0, focusthickness=0)
        s.map("Primary.TButton", background=[("active", "#258f8a"), ("disabled", "#18373c")])
        s.configure("Selected.TButton", font=("Segoe UI", 9, "bold"), padding=(10, 7), background="#164744", foreground="#a8ece1", relief="flat", borderwidth=0, focusthickness=0)
        s.map("Selected.TButton", background=[("active", "#1c5f5b")])
        s.configure("Treeview", background="#111a22", foreground=TEXT, fieldbackground="#111a22", rowheight=30, borderwidth=0, relief="flat", font=("Segoe UI", 9), bordercolor="#111a22", lightcolor="#111a22", darkcolor="#111a22", focuscolor="#111a22")
        s.configure("Treeview.Heading", background=PANEL2, foreground="#aabac6", font=("Segoe UI", 8, "bold"), relief="flat", borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
        s.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        s.map("Treeview", background=[("selected", "#214b50")], foreground=[("selected", "#efffff")])
        s.configure("TEntry", fieldbackground="#0e1720", foreground=TEXT, insertcolor=TEXT, padding=(7, 6), borderwidth=0, relief="flat")
        s.configure("TCombobox", fieldbackground="#0e1720", foreground=TEXT, padding=(7, 6), borderwidth=0, relief="flat", arrowsize=12)
        s.configure("TCheckbutton", background=PANEL, foreground=TEXT, font=("Segoe UI", 9))

    def _build(self) -> None:
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)
        self.nav = tk.Frame(shell, bg=SIDEBAR, width=194)
        self.nav.pack(side="left", fill="y")
        self.nav.pack_propagate(False)
        brand = tk.Frame(self.nav, bg=SIDEBAR); brand.pack(fill="x", padx=16, pady=(20, 24))
        mark = tk.Canvas(brand, width=28, height=28, bg=SIDEBAR, highlightthickness=0)
        mark.create_oval(3, 3, 25, 25, outline=ACCENT, width=2)
        mark.create_text(14, 14, text="F", fill=ACCENT, font=("Segoe UI", 12, "bold"))
        mark.pack(side="left", padx=(0, 9))
        tk.Label(brand, text="FireSight", bg=SIDEBAR, fg=TEXT, font=("Segoe UI", 15, "bold")).pack(side="left", anchor="center")
        tk.Label(self.nav, text="OPERATIONS", bg=SIDEBAR, fg="#738392", font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=19, pady=(0, 8))
        self.page_buttons = {}
        nav_items = (
            (("◉", "Live Monitor"), "monitor", lambda: self._show_page("monitor")),
            (("○", "Dashboard / Analytics"), "analytics", lambda: self._show_page("analytics")),
            (("⌕", "Camera Sources"), "sources", self.show_cctv),
            (("!", "Alerts"), "alerts", self.show_alerts),
            (("≡", "Events & Evidence"), "events", lambda: self._show_page("events")),
            (("◇", "Settings"), "settings", lambda: self._show_page("settings")),
            (("＋", "System Health"), "health", lambda: self._show_page("health")),
        )
        self.nav_labels = {}
        for (icon, label), page, command in nav_items:
            b = tk.Button(self.nav, text=f"{icon}   {label}", anchor="w", relief="flat", bd=0, bg=SIDEBAR, fg="#b6c3cc", activebackground=PANEL2, activeforeground=TEXT, font=("Segoe UI", 9), padx=13, pady=10, command=command, cursor="hand2")
            b.pack(fill="x", padx=8, pady=2); self.page_buttons[page] = b; self.nav_labels[page] = label
            b.bind("<Enter>", lambda _e, button=b, key=page: button.configure(bg="#17242d" if key != getattr(self, "_active_page", "monitor") else "#164744"))
            b.bind("<Leave>", lambda _e, button=b, key=page: button.configure(bg="#164744" if key == getattr(self, "_active_page", "monitor") else SIDEBAR))

        self.content = tk.Frame(shell, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)
        self.header = tk.Frame(self.content, bg=BG)
        self.header.pack(fill="x", padx=24, pady=(16, 12))
        self.page_title_var = StringVar(value="Live Monitor")
        self.header_source_var = StringVar(value="No source selected")
        self.status_var = StringVar(value="INITIALIZING")
        title = tk.Frame(self.header, bg=BG); title.pack(side="left")
        tk.Label(title, textvariable=self.page_title_var, bg=BG, fg=TEXT, font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(title, textvariable=self.header_source_var, bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))
        right_header = tk.Frame(self.header, bg=BG); right_header.pack(side="right")
        self.clock_var = StringVar(value="")
        tk.Label(right_header, textvariable=self.clock_var, bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right", padx=(14, 0))
        self.status_chip = tk.Label(right_header, textvariable=self.status_var, bg=PANEL2, fg=TEXT, font=("Segoe UI", 8, "bold"), padx=10, pady=6)
        self.status_chip.pack(side="right")

        self.page_host = tk.Frame(self.content, bg=BG)
        self.page_host.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.pages = {}
        self._build_monitor()
        self._build_sources_page()
        self._build_alerts_page()
        self._build_events_page()
        self._build_analytics_page()
        self._build_settings_page()
        self._build_health_page()
        self._show_page("monitor")
        self.refresh_events()
        self._update_clock()

    def _card(self, parent, title: str, value: str = "—", color: str = TEXT):
        f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#233342")
        tk.Label(f, text=title.upper(), bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=13, pady=(11, 3))
        v = tk.Label(f, text=value, bg=PANEL, fg=color, font=("Segoe UI", 14, "bold"), anchor="w")
        v.pack(fill="x", padx=13, pady=(0, 11))
        return f, v

    def _build_monitor(self) -> None:
        page = tk.Frame(self.page_host, bg=BG); self.pages["monitor"] = page
        self.alert_banner = tk.Frame(page, bg="#17232a", height=32)
        self.alert_banner.pack(fill="x", pady=(0, 9)); self.alert_banner.pack_propagate(False)
        self.alert_var = StringVar(value="MONITORING READY  ·  No confirmed incidents")
        self.alert_label = tk.Label(self.alert_banner, textvariable=self.alert_var, bg="#17232a", fg=MUTED, font=("Segoe UI", 9, "bold"))
        self.alert_label.pack(side="left", padx=12, pady=6)

        cards = tk.Frame(page, bg=BG); cards.pack(fill="x", pady=(0, 10))
        for i in range(3): cards.columnconfigure(i, weight=1, uniform="status")
        self.camera_card, self.camera_value = self._card(cards, "Camera status", "No source selected", MUTED)
        self.fps_card, self.fps_value = self._card(cards, "Live performance", "— FPS   ·   — resolution", MUTED)
        self.detection_card, self.detection_value = self._card(cards, "Detection summary · current frame", "Fire 0   ·   Smoke 0   ·   Person 0", MUTED)
        for i, card in enumerate((self.camera_card, self.fps_card, self.detection_card)):
            card.configure(highlightthickness=0)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0 if i == 2 else 6))

        self.save_video = BooleanVar(value=False)
        self.camera_name_var = StringVar(value=self.config.camera_name)
        self.rtsp_url_var = StringVar(value=sanitize_source(self.config.rtsp_url))
        self.username_var = StringVar(); self.password_var = StringVar()
        self.cctv_error = StringVar(value="")
        self.camera_selector_var = StringVar(value=self.config.camera_name or "Camera 1")

        workspace = tk.Frame(page, bg=BG); workspace.pack(fill="both", expand=True, pady=(0, 10))
        workspace.columnconfigure(0, weight=68, uniform="monitor")
        workspace.columnconfigure(1, weight=32, uniform="monitor")
        workspace.rowconfigure(0, weight=1)
        video_card = tk.Frame(workspace, bg="#070c10", highlightthickness=1, highlightbackground="#22313b")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12)); video_card.rowconfigure(0, weight=1); video_card.columnconfigure(0, weight=1)
        self.feed = tk.Label(video_card, text="", bg="#070c10", fg=MUTED, font=("Segoe UI", 11), compound="center")
        self.feed.grid(row=0, column=0, sticky="nsew")
        self.feed_empty = tk.Frame(video_card, bg="#070c10")
        self.feed_empty.place(relx=.5, rely=.5, anchor="center")
        tk.Label(self.feed_empty, text="◉", bg="#070c10", fg="#31525a", font=("Segoe UI Symbol", 30)).pack(pady=(0, 8))
        tk.Label(self.feed_empty, text="No live feed", bg="#070c10", fg="#d2dce3", font=("Segoe UI", 13, "bold")).pack()
        tk.Label(self.feed_empty, text="Select a source to begin monitoring", bg="#070c10", fg=MUTED, font=("Segoe UI", 9)).pack(pady=(4, 0))
        self.feed_badge = tk.Label(video_card, text="", bg="#0e191f", fg=GREEN, font=("Segoe UI", 8, "bold"), padx=9, pady=5)
        self.feed_badge.place(x=12, y=12)
        self.feed_caption = tk.Label(video_card, text="NO ACTIVE SOURCE", anchor="w", bg="#0e171e", fg=MUTED, font=("Segoe UI", 8), padx=11, pady=7)
        self.feed_caption.grid(row=1, column=0, sticky="ew")

        right = tk.Frame(workspace, bg=PANEL, padx=15, pady=13)
        right.grid(row=0, column=1, sticky="nsew")
        tk.Label(right, text="SOURCE", bg=PANEL, fg="#aebcc6", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 8))
        self.source_buttons = {}
        source_row = tk.Frame(right, bg=PANEL); source_row.pack(fill="x")
        source_options = (("Webcam", "webcam"), ("CCTV / RTSP", "cctv"), ("Video", "video"), ("Image", "image"))
        for index, (label, key) in enumerate(source_options):
            button = tk.Button(source_row, text=label, relief="flat", bd=0, bg=PANEL2, fg=TEXT, activebackground="#24414a", activeforeground=TEXT, font=("Segoe UI", 9, "bold"), padx=8, pady=9, cursor="hand2", command=lambda source=key: self._select_source(source))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0, 5) if index % 2 == 0 else (5, 0), pady=(0, 7))
            source_row.columnconfigure(index % 2, weight=1, uniform="source")
            self.source_buttons[key] = button
        self.source_action_panel = tk.Frame(right, bg=PANEL)
        self.source_action_panel.pack(fill="x", pady=(2, 0))
        self.source_hint_var = StringVar(value="Select an input source.")
        self.source_hint = tk.Label(self.source_action_panel, textvariable=self.source_hint_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), justify="left", wraplength=320)
        self.source_hint.pack(anchor="w", pady=(0, 7))
        self.stop_button = ttk.Button(self.source_action_panel, text="STOP SOURCE", command=self.stop_source, state="disabled")
        self.stop_button.pack_forget()
        ttk.Checkbutton(right, text="Save processed video", variable=self.save_video).pack(anchor="w", pady=(5, 12))
        tk.Frame(right, bg="#263640", height=1).pack(fill="x", pady=(0, 10))
        tk.Label(right, text="LATEST DETECTION", bg=PANEL, fg="#aebcc6", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 6))
        self.detection_detail_var = StringVar(value="Waiting for a frame")
        tk.Label(right, textvariable=self.detection_detail_var, bg=PANEL, fg=TEXT, font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w")
        self.detection_time_var = StringVar(value="No feed timestamp")
        tk.Label(right, textvariable=self.detection_time_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))

        recent = tk.Frame(page, bg=PANEL, padx=11, pady=9)
        recent.pack(fill="x")
        recent_header = tk.Frame(recent, bg=PANEL); recent_header.pack(fill="x", pady=(0, 6))
        tk.Label(recent_header, text="Recent Events", bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(recent_header, text="VIEW ALL EVENTS", command=lambda: self._show_page("events")).pack(side="right")
        self.event_tree = self._make_event_tree(recent, compact=True)
        self.event_tree.pack(fill="x")
        recent_actions = tk.Frame(recent, bg=PANEL); recent_actions.pack(fill="x", pady=(6, 0))
        tk.Label(recent_actions, text="Double-click an event or select it to inspect evidence.", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        ttk.Button(recent_actions, text="VIEW EVIDENCE", command=lambda: self.view_selected_evidence(self.event_tree)).pack(side="right")
        initial_source = "cctv" if self.config.rtsp_url else "webcam"
        self._select_source(initial_source)

    def _field(self, parent, label, variable, *, secret=False, hint=None):
        tk.Label(parent, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(5, 2))
        entry = ttk.Entry(parent, textvariable=variable, show="•" if secret else "")
        entry.pack(fill="x")
        if hint:
            tk.Label(parent, text=hint, bg=PANEL, fg="#71859a", font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

    def _make_event_tree(self, parent, compact=False):
        columns = ("time", "camera", "event", "conf", "severity", "status", "evidence")
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse", height=4 if compact else 15)
        widths = {"time": 125 if not compact else 120, "camera": 180 if not compact else 175, "event": 105 if not compact else 90, "conf": 80 if not compact else 76, "severity": 105 if not compact else 96, "status": 125 if not compact else 115, "evidence": 85 if not compact else 80}
        labels = {"time":"TIME", "camera":"CAMERA", "event":"EVENT", "conf":"CONF.", "severity":"SEVERITY", "status":"STATUS", "evidence":"EVIDENCE"}
        for col in columns:
            label = labels[col]
            tree.heading(col, text=label); tree.column(col, width=widths[col], stretch=col in {"time", "camera"}, anchor="center" if col in {"conf", "severity", "status", "evidence"} else "w")
        tree.bind("<Double-1>", lambda _e: self.view_selected_evidence(tree))
        return tree

    def _build_events_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["events"] = page
        tk.Label(page, text="Search the complete confirmed incident history and open its saved evidence.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 10))
        filters = tk.Frame(page, bg=PANEL, padx=12, pady=10); filters.pack(fill="x", pady=(0, 10))
        self.event_search = StringVar(); self.event_type_filter = StringVar(value="All types"); self.event_status_filter = StringVar(value="All statuses"); self.event_camera_filter = StringVar(value="All cameras")
        self.event_after = StringVar(); self.event_before = StringVar()
        self.event_sort = StringVar(value="Newest first")
        self.event_severity_filter = StringVar(value="All severities")
        filter_row = tk.Frame(filters, bg=PANEL); filter_row.pack(fill="x")
        ttk.Entry(filter_row, textvariable=self.event_search, width=24).pack(side="left", padx=(0, 6))
        ttk.Combobox(filter_row, textvariable=self.event_type_filter, values=("All types", "Fire", "Smoke"), state="readonly", width=10).pack(side="left", padx=3)
        self.event_camera_box = ttk.Combobox(filter_row, textvariable=self.event_camera_filter, values=("All cameras",), state="readonly", width=14); self.event_camera_box.pack(side="left", padx=3)
        self.event_severity_box = ttk.Combobox(filter_row, textvariable=self.event_severity_filter, values=("All severities",), state="readonly", width=12); self.event_severity_box.pack(side="left", padx=3)
        ttk.Combobox(filter_row, textvariable=self.event_status_filter, values=("All statuses", "Acknowledged", "Unacknowledged"), state="readonly", width=14).pack(side="left", padx=3)
        ttk.Combobox(filter_row, textvariable=self.event_sort, values=("Newest first", "Oldest first"), state="readonly", width=12).pack(side="left", padx=3)
        dates_row = tk.Frame(filters, bg=PANEL); dates_row.pack(fill="x", pady=(7, 0))
        tk.Label(dates_row, text="From", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left", padx=(0, 3))
        ttk.Entry(dates_row, textvariable=self.event_after, width=14).pack(side="left", padx=(0, 10))
        tk.Label(dates_row, text="To", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left", padx=(0, 3))
        ttk.Entry(dates_row, textvariable=self.event_before, width=14).pack(side="left")
        tk.Label(dates_row, text="Date format  YYYY-MM-DD", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(side="left", padx=9)
        ttk.Button(dates_row, text="APPLY FILTERS", command=self._apply_event_filters).pack(side="right")
        card = tk.Frame(page, bg=PANEL, padx=10, pady=10); card.pack(fill="both", expand=True)
        self.history_tree = self._make_event_tree(card); self.history_tree.pack(fill="both", expand=True)
        actions = tk.Frame(page, bg=BG); actions.pack(fill="x", pady=10)
        ttk.Button(actions, text="ACKNOWLEDGE SELECTED", command=lambda: self.acknowledge_selected_event(self.history_tree)).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="VIEW SELECTED EVIDENCE", command=lambda: self.view_selected_evidence(self.history_tree)).pack(side="right")
        self._event_page = 0
        self.event_page_info = StringVar(value="")
        pager = tk.Frame(page, bg=BG); pager.pack(fill="x", pady=(0, 8))
        ttk.Button(pager, text="NEWER", command=lambda: self._change_event_page(-1)).pack(side="right", padx=(5, 0))
        ttk.Button(pager, text="OLDER", command=lambda: self._change_event_page(1)).pack(side="right")
        tk.Label(pager, textvariable=self.event_page_info, bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="right", padx=12)

    def _build_alerts_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["alerts"] = page
        tk.Label(page, text="Alert management", bg=BG, fg=TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 5))
        tk.Label(page, text="Acknowledgement records operator review. Silencing mutes the local alarm; neither action resolves an incident.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 12))
        card = tk.Frame(page, bg=PANEL, padx=10, pady=10); card.pack(fill="both", expand=True)
        self.alert_tree = self._make_event_tree(card); self.alert_tree.pack(fill="both", expand=True)
        actions = tk.Frame(page, bg=BG); actions.pack(fill="x", pady=10)
        self.ack_button = ttk.Button(actions, text="ACKNOWLEDGE ACTIVE ALERT", command=self.acknowledge_alert, state="disabled"); self.ack_button.pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="ACKNOWLEDGE SELECTED", command=lambda: self.acknowledge_selected_event(self.alert_tree)).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="VIEW EVIDENCE", command=lambda: self.view_selected_evidence(self.alert_tree)).pack(side="right", padx=6)
        ttk.Button(actions, text="SILENCE LOCAL ALARM", command=self.silence_alarm).pack(side="left")

    def _build_settings_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["settings"] = page
        tk.Label(page, text="Tune local detection and alert behavior. External delivery is disabled and no sender is configured.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 14))
        card = tk.Frame(page, bg=PANEL, padx=18, pady=14); card.pack(fill="x", anchor="n")
        self.setting_conf = StringVar(value=str(self.config.confidence_threshold))
        self.setting_frames = StringVar(value=str(self.config.confirmation_frames))
        self.setting_model = StringVar(value=str(self.config.model_path))
        self.setting_source = StringVar(value=self.config.video_source)
        self.setting_rtsp = StringVar(value=sanitize_source(self.config.rtsp_url))
        self.setting_camera = StringVar(value=self.config.camera_name)
        self.setting_evidence = StringVar(value=str(self.config.evidence_dir))
        self.alarm_enabled_var = BooleanVar(value=self.config.alarm_enabled)
        self.external_enabled_var = BooleanVar(value=self.config.external_alerts_enabled)
        self.email_alert_var = BooleanVar(value=self.config.email_alerts_enabled)
        self.telegram_alert_var = BooleanVar(value=self.config.telegram_alerts_enabled)
        self.webhook_alert_var = BooleanVar(value=self.config.webhook_alerts_enabled)
        groups = (
            ("MODEL & DETECTION", (("Model weights path · applies next launch", self.setting_model), ("Confidence threshold · 0.0–1.0", self.setting_conf), ("Consecutive confirmation frames", self.setting_frames))),
            ("DEFAULT SOURCE", (("Video/camera source", self.setting_source), ("RTSP endpoint · credentials excluded", self.setting_rtsp), ("Camera name", self.setting_camera))),
            ("EVIDENCE STORAGE", (("Evidence directory", self.setting_evidence),)),
        )
        for group_name, fields in groups:
            section = tk.Frame(card, bg=PANEL); section.pack(fill="x", pady=(0, 8))
            tk.Label(section, text=group_name, bg=PANEL, fg=ACCENT, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(6, 1))
            tk.Frame(section, bg="#263640", height=1).pack(fill="x", pady=(0, 4))
            for label, variable in fields: self._setting_field(section, label, variable)
        ttk.Checkbutton(card, text="Enable local audible alarm", variable=self.alarm_enabled_var).pack(anchor="w", pady=(12, 4))
        ttk.Checkbutton(card, text="Enable external alert configuration (future integration only)", variable=self.external_enabled_var).pack(anchor="w", pady=4)
        channels = tk.Frame(card, bg=PANEL); channels.pack(anchor="w", pady=(3, 4))
        for title, variable in (("Email", self.email_alert_var), ("Telegram", self.telegram_alert_var), ("Webhook", self.webhook_alert_var)):
            ttk.Checkbutton(channels, text=title, variable=variable).pack(side="left", padx=(0, 12))
        tk.Label(card, text="External alert delivery is not implemented. These selections are saved for future setup only; no external notifications are sent.", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), wraplength=850, justify="left").pack(anchor="w", pady=(6, 12))
        ttk.Button(card, text="SAVE SETTINGS", command=self.save_settings).pack(anchor="w")

    def _build_sources_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["sources"] = page
        tk.Label(page, text="Camera Sources", bg=BG, fg=TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 4))
        tk.Label(page, text="Configure and test RTSP cameras. Credentials are used in memory and are never saved or shown in source URLs.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 14))
        card = tk.Frame(page, bg=PANEL, padx=18, pady=16); card.pack(fill="x", anchor="n")
        self._field(card, "Camera name", self.camera_name_var)
        self._field(card, "RTSP URL", self.rtsp_url_var, hint="rtsp://camera-address:554/stream — do not include credentials")
        credentials = tk.Frame(card, bg=PANEL); credentials.pack(fill="x", pady=(0, 2)); credentials.columnconfigure((0,1), weight=1)
        self._field(credentials, "Username", self.username_var)
        self._field(credentials, "Password", self.password_var, secret=True)
        tk.Label(card, text="Connection type  ·  RTSP", bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 4))
        controls = tk.Frame(card, bg=PANEL); controls.pack(fill="x", pady=8)
        ttk.Button(controls, text="CONNECT CAMERA", command=self.connect_cctv).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="TEST CONNECTION", command=self.test_cctv).pack(side="left", padx=6)
        ttk.Button(controls, text="DISCONNECT", command=self.stop_source).pack(side="left", padx=6)
        self.reconnect_button = ttk.Button(controls, text="RECONNECT", command=self.reconnect_cctv, state="disabled"); self.reconnect_button.pack(side="left", padx=6)
        tk.Label(card, textvariable=self.cctv_error, bg=PANEL, fg="#ff8f8f", font=("Segoe UI", 9), wraplength=900, justify="left").pack(anchor="w", pady=(4, 0))

    def _build_health_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["health"] = page
        tk.Label(page, text="System Health", bg=BG, fg=TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 12))
        card = tk.Frame(page, bg=PANEL, padx=18, pady=16); card.pack(fill="x", anchor="n")
        self.health_model = StringVar(value="Loading model…")
        self.health_camera = StringVar(value="No active source")
        self.health_frames = StringVar(value="No frames processed")
        self.health_storage = StringVar(value=self._storage_summary())
        self.health_logs = StringVar(value=self._diagnostic_paths_summary())
        for label, var in (("MODEL", self.health_model), ("CAMERA", self.health_camera), ("FRAME PROCESSING", self.health_frames), ("STORAGE AVAILABILITY", self.health_storage), ("EVENT DATABASE & DETECTION LOG", self.health_logs)):
            tk.Label(card, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(5, 0))
            tk.Label(card, textvariable=var, bg=PANEL, fg=TEXT, font=("Segoe UI", 10), wraplength=900, justify="left").pack(anchor="w", pady=(2, 8))

    def _build_analytics_page(self):
        page = tk.Frame(self.page_host, bg=BG); self.pages["analytics"] = page
        tk.Label(page, text="Historical overview", bg=BG, fg=TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 5))
        tk.Label(page, text="Summaries use confirmed incidents stored in the local event history.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 12))
        card = tk.Frame(page, bg=PANEL, padx=18, pady=16); card.pack(fill="x", anchor="n")
        self.analytics_total = StringVar(value="0")
        self.analytics_fire = StringVar(value="0")
        self.analytics_smoke = StringVar(value="0")
        self.analytics_cameras = StringVar(value="No recorded events")
        self.analytics_trend = StringVar(value="No events recorded in the last 7 days")
        self.analytics_health = StringVar(value="No active source")
        self.analytics_recent = StringVar(value="No recorded events")
        for label, var in (("CONFIRMED INCIDENTS", self.analytics_total), ("FIRE EVENTS", self.analytics_fire), ("SMOKE EVENTS", self.analytics_smoke), ("CAMERAS WITH RECORDED EVENTS", self.analytics_cameras), ("LAST 7 DAYS · EVENT TREND (UTC)", self.analytics_trend), ("CURRENT CAMERA STATE", self.analytics_health), ("RECENT ACTIVITY", self.analytics_recent)):
            tk.Label(card, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(5, 0))
            tk.Label(card, textvariable=var, bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold"), wraplength=900, justify="left").pack(anchor="w", pady=(2, 8))

    def _storage_summary(self):
        location = self.config.evidence_dir.expanduser()
        try:
            probe = location if location.exists() else location.parent
            free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
            available = "Available" if location.exists() and location.is_dir() else "Directory will be created when evidence is saved"
            return f"{available} · {free_gb:.1f} GB free · {location}"
        except OSError as exc:
            return f"Unavailable · {exc} · {location}"

    def _diagnostic_paths_summary(self):
        parts = []
        for label, path in (("Event database", self.config.event_db), ("Detection log", self.config.detection_log)):
            try:
                size = path.stat().st_size if path.is_file() else 0
                state = f"present · {size:,} bytes" if path.is_file() else "will be created when needed"
            except OSError as exc:
                state = f"unavailable · {exc}"
            parts.append(f"{label}: {state} · {path}")
        return "\n".join(parts)

    def _setting_field(self, parent, label, variable):
        row = tk.Frame(parent, bg=PANEL); row.pack(fill="x", pady=5)
        tk.Label(row, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 9, "bold"), width=32, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

    def _show_page(self, page):
        for frame in self.pages.values(): frame.pack_forget()
        self.pages[page].pack(fill="both", expand=True)
        self.page_title_var.set({"monitor":"Live Monitor", "sources":"Camera Sources", "alerts":"Alerts", "events":"Events & Evidence", "analytics":"Dashboard / Analytics", "settings":"Settings", "health":"System Health"}[page])
        self._active_page = page
        for key, button in self.page_buttons.items():
            active = key == page
            button.configure(bg="#164744" if active else SIDEBAR, fg="#d4fff8" if active else "#b6c3cc")
        if page in {"events", "alerts"}: self.refresh_events()
        if page == "analytics": self.refresh_events()

    def _update_clock(self):
        if self._closed: return
        self.clock_var.set(time.strftime("%H:%M:%S"))
        self.root.after(1000, self._update_clock)

    def _select_source(self, source):
        self._selected_source = source
        self._mark_source(source)
        source_names = {"webcam": "Webcam", "cctv": self.camera_name_var.get().strip() or "CCTV / RTSP", "video": "Video file", "image": "Image file"}
        if not self.busy.is_set(): self.header_source_var.set(f"{source_names[source]} · selected")
        if not self.busy.is_set() and not self._connected:
            state = f"{source_names[source]}  ·  DISCONNECTED" if source in {"webcam", "cctv"} else f"{source_names[source]}  ·  READY"
            self.camera_value.configure(text=state, fg=MUTED)
        hint = {
            "webcam": "Use the default local webcam for live monitoring.",
            "cctv": "Select a configured RTSP camera. Manage endpoints and credentials in Camera Sources.",
            "video": "Choose a local video file to run detection and preview annotated frames.",
            "image": "Choose a local image to run a single detection pass.",
        }[source]
        self.source_hint_var.set(hint)
        for child in self.source_action_panel.winfo_children():
            if child not in (self.source_hint, self.stop_button): child.destroy()
        if source == "webcam":
            ttk.Button(self.source_action_panel, text="START WEBCAM", style="Primary.TButton", command=self.start_webcam).pack(fill="x", pady=3)
        elif source == "cctv":
            self.camera_selector = ttk.Combobox(self.source_action_panel, textvariable=self.camera_selector_var, values=(self.config.camera_name or "Camera 1",), state="readonly")
            self.camera_selector.pack(fill="x", pady=(2, 6))
            ttk.Button(self.source_action_panel, text="CONNECT CAMERA", style="Primary.TButton", command=self.connect_camera_from_monitor).pack(fill="x", pady=3)
            ttk.Button(self.source_action_panel, text="MANAGE CAMERA SOURCES", command=self.show_cctv).pack(fill="x", pady=3)
        elif source == "video":
            ttk.Button(self.source_action_panel, text="CHOOSE VIDEO FILE", style="Primary.TButton", command=self.open_video).pack(fill="x", pady=3)
        else:
            ttk.Button(self.source_action_panel, text="CHOOSE IMAGE", style="Primary.TButton", command=self.open_image).pack(fill="x", pady=3)

    def connect_camera_from_monitor(self):
        selected_name = self.camera_selector_var.get().strip()
        if selected_name: self.camera_name_var.set(selected_name)
        if not self.rtsp_url_var.get().strip():
            self.show_cctv()
            self.cctv_error.set("Add an RTSP endpoint in Camera Sources before connecting.")
            return
        self.connect_cctv()

    def _apply_event_filters(self):
        try:
            if self.event_after.get().strip(): date.fromisoformat(self.event_after.get().strip())
            if self.event_before.get().strip(): date.fromisoformat(self.event_before.get().strip())
        except ValueError:
            messagebox.showerror("Invalid date", "Enter dates as YYYY-MM-DD.", parent=self.root)
            return
        self._event_page = 0
        self.refresh_events()

    def _change_event_page(self, delta):
        self._event_page = max(0, self._event_page + delta)
        self.refresh_events()

    def show_cctv(self):
        if self.busy.is_set():
            self._pending_action = self.show_cctv
            self.stop_event.set()
            self._status("Switching source · releasing current camera", AMBER)
            return
        self._show_page("sources")
        self._mark_source("cctv")
        self._camera_name = self.camera_name_var.get().strip() or "CCTV Camera"
        self._source_type = "CCTV / RTSP"
        if not self.busy.is_set(): self._camera_state("OFFLINE", camera_name=self._camera_name, source_type=self._source_type)

    def show_alerts(self):
        self._show_page("alerts")

    def _status(self, text: str, color: str = PANEL2):
        self.status_var.set(text.upper()); self.status_chip.configure(bg=color)

    def _load_model(self):
        try:
            names = self.engine.load_model(); self.messages.put(("model", names))
        except Exception as exc: self.messages.put(("error", f"Model load failed: {exc}"))

    def _idle(self, retry_action=None):
        if self.busy.is_set():
            if retry_action is not None:
                self._pending_action = retry_action
                self.stop_event.set()
                self._status("Switching source · releasing current camera", AMBER)
            return False
        if self.engine.model is None:
            messagebox.showwarning("Model loading", "Wait for the trained model to finish loading.", parent=self.root); return False
        return True

    def _begin(self, target, name):
        self.busy.set(); self.stop_event.clear(); self.stop_button.configure(state="normal")
        if name != "firesight-image" and not self.stop_button.winfo_manager(): self.stop_button.pack(fill="x", pady=(5, 0))
        self._status("Processing", GREEN)
        self._worker_thread = threading.Thread(target=target, name=name, daemon=True); self._worker_thread.start()

    def open_image(self):
        if not self._idle(self.open_image): return
        path = filedialog.askopenfilename(parent=self.root, title="Select image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        if not path: return
        self._mark_source("image")
        self._camera_name = Path(path).name; self._source_type = "IMAGE"
        self.header_source_var.set(f"{self._camera_name} · IMAGE · processing")
        self.camera_value.configure(text=f"{self._camera_name}  ·  PROCESSING", fg=ACCENT)
        self.feed_empty.place_forget()
        self.feed_caption.configure(text=f"IMAGE  ·  {Path(path).name}")
        def work():
            try:
                result, output = self.engine.process_image(path); self._post_frame(result); self.messages.put(("image_done", str(output), result))
            except Exception as exc: self.messages.put(("error", f"Image inference failed: {exc}"))
            finally: self.busy.clear(); self.messages.put(("idle",))
        self._begin(work, "firesight-image")

    def open_video(self):
        if not self._idle(self.open_video): return
        path = filedialog.askopenfilename(parent=self.root, title="Select video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if path: self._start_video(path, camera_name=Path(path).stem)

    def start_webcam(self):
        if not self._idle(self.start_webcam): return
        self._mark_source("webcam")
        self._camera_name = "Laptop Webcam"; self._source_type = "WEBCAM"
        self._start_video(0, camera_name=self._camera_name)

    def connect_cctv(self):
        if not self._idle(self.connect_cctv): return
        self._camera_name = self.camera_name_var.get().strip() or "CCTV Camera"; self._source_type = "CCTV / RTSP"
        try:
            source = build_rtsp_url(self.rtsp_url_var.get(), self.username_var.get(), self.password_var.get())
        except ValueError as exc:
            self.cctv_error.set(str(exc)); self._camera_state("ERROR", str(exc), True); return
        self.cctv_error.set("")
        self._rtsp_source = source
        self._start_video(source, camera_name=self._camera_name, is_cctv=True)

    def reconnect_cctv(self):
        self.connect_cctv()

    def test_cctv(self):
        if self.busy.is_set():
            self.cctv_error.set("Stop the active source before testing another camera connection."); return
        try:
            source = build_rtsp_url(self.rtsp_url_var.get(), self.username_var.get(), self.password_var.get())
        except ValueError as exc:
            self.cctv_error.set(str(exc)); return
        self.cctv_error.set("Testing connection…")
        self.reconnect_button.configure(state="disabled")
        def probe():
            try:
                width, height = test_video_connection(source)
                self.messages.put(("probe_done", f"Connection test passed · {width} × {height}"))
            except Exception:
                self.messages.put(("probe_done", "Unable to connect. Check RTSP URL, network access, and camera credentials."))
        threading.Thread(target=probe, name="firesight-rtsp-probe", daemon=True).start()

    def _start_video(self, source, *, camera_name, is_cctv=False):
        self._mark_source("cctv" if is_cctv else "webcam" if isinstance(source, int) else "video")
        safe_source = sanitize_source(source)
        self._current_source = (source, camera_name, is_cctv)
        self._source_type = "CCTV / RTSP" if is_cctv else "WEBCAM" if isinstance(source, int) else "VIDEO"
        self._camera_name = camera_name
        self._frames_processed = 0; self._last_frame_monotonic = None; self._resolution = "—"; self._frame_times.clear()
        self._reconnect_attempts = 0
        self.fps_value.configure(text="— FPS  ·  —")
        self._counts = {"fire": 0, "smoke": 0, "person": 0}
        self.detection_value.configure(text="Fire 0   ·   Smoke 0   ·   Person 0")
        self.feed_caption.configure(text=f"CONNECTING  ·  {camera_name}  ·  {safe_source}")
        self.header_source_var.set(f"{camera_name} · {self._source_type}")
        self._camera_state("CONNECTING", None, is_cctv)
        self._status("Connecting camera…", AMBER)
        def on_frame(result):
            self._post_frame(result)
            if result.events: self.messages.put(("events", result.events))
            self.messages.put(("health", result.timestamp))
        def on_status(state, error): self.messages.put(("camera", state, error, is_cctv, camera_name, self._source_type))
        def work():
            try:
                summary = run_video_source(self.engine, source, stop_event=self.stop_event, save_output=self.save_video.get(), on_frame=on_frame, camera_name=camera_name, on_status=on_status)
                self.messages.put(("video_done", summary.frames_processed, str(summary.output_path) if summary.output_path else None))
            except Exception as exc:
                safe_error = str(exc).replace(str(source), safe_source)
                self.messages.put(("camera", "ERROR", safe_error, is_cctv, camera_name, self._source_type))
                if not self.stop_event.is_set(): self.messages.put(("error", safe_error, is_cctv))
            finally:
                self.busy.clear(); self.messages.put(("idle",))
        self._begin(work, "firesight-stream")

    def _mark_source(self, selected):
        for key, button in self.source_buttons.items():
            button.configure(bg="#15524f" if key == selected else PANEL2, fg="#b9f2e8" if key == selected else TEXT)

    def _camera_state(self, state, error=None, is_cctv=False, camera_name=None, source_type=None):
        status = state.upper()
        self._connected = status == "CONNECTED"
        color = GREEN if status == "CONNECTED" else AMBER if "CONNECT" in status else CRITICAL if status == "ERROR" else MUTED
        self.camera_value.configure(text=f"{camera_name or self._camera_name}  ·  {status}", fg=color)
        if camera_name: self._camera_name = camera_name
        if source_type: self._source_type = source_type
        self.header_source_var.set(f"{self._camera_name} · {self._source_type}")
        self.feed_caption.configure(text=f"{self._camera_name}  ·  {status}")
        self.feed_badge.configure(text="● LIVE" if status == "CONNECTED" else "● RECONNECTING" if status.startswith("RECONNECTING") else "", fg=GREEN if status == "CONNECTED" else AMBER)
        if status != "CONNECTED":
            text = "RECONNECTING TO SOURCE" if status.startswith("RECONNECTING") else "CONNECTING TO SOURCE" if status == "CONNECTING" else ""
            self.feed.configure(image="", text=text, fg=AMBER if "CONNECT" in status else MUTED)
            self._photo = None
            self.fps_value.configure(text="— FPS  ·  —")
            self.detection_value.configure(text="No active feed")
            if status.startswith("RECONNECTING"):
                self.feed_empty.place_forget()
            elif status != "CONNECTING":
                self.feed_empty.place(relx=.5, rely=.5, anchor="center")
        if is_cctv and self._reconnect_attempts:
            self.feed_caption.configure(text=f"{self._camera_name}  ·  {status}  ·  {self._reconnect_attempts} retries")
        show_reconnect = is_cctv and status in {"ERROR", "OFFLINE"}
        self.reconnect_button.configure(state="normal" if show_reconnect else "disabled")
        if show_reconnect and not self.reconnect_button.winfo_manager(): self.reconnect_button.pack(side="left", padx=6)
        elif not show_reconnect and self.reconnect_button.winfo_manager(): self.reconnect_button.pack_forget()
        if status == "ERROR":
            self.feed.configure(text="CONNECTION ERROR", fg=CRITICAL)
            self.feed_empty.place_forget()
        if error and is_cctv: self.cctv_error.set(error)

    def _camera_offline(self):
        self._camera_state("OFFLINE", camera_name=self._camera_name, source_type=self._source_type)

    def stop_source(self):
        self.cctv_error.set("")
        if self.busy.is_set(): self.stop_event.set(); self._status("Disconnecting…", AMBER)
        else: self._camera_offline()

    def _post_frame(self, result):
        try: self.frames.put_nowait(result)
        except queue.Full:
            try: self.frames.get_nowait()
            except queue.Empty: pass
            self.frames.put_nowait(result)

    def _render_frame(self, result):
        rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb); image.thumbnail((max(360, self.feed.winfo_width()), max(260, self.feed.winfo_height())), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=image); self.feed.configure(image=self._photo, text="")
        self.feed_empty.place_forget()
        self._frame_times.append(time.monotonic()); self._frame_times = self._frame_times[-30:]
        self._last_frame_monotonic = time.monotonic(); self._frames_processed += 1
        height, width = result.annotated_frame.shape[:2]
        self._resolution = f"{width} × {height}"
        fps = (len(self._frame_times)-1)/(self._frame_times[-1]-self._frame_times[0]) if len(self._frame_times)>1 else 0
        self.fps_value.configure(text=f"{fps:.1f} FPS  ·  {self._resolution}")
        current = {name: 0 for name in ("fire", "smoke", "person")}
        for detection in result.detections:
            if detection.class_name in current: current[detection.class_name] += 1
        self._counts = current
        self.detection_value.configure(text=f"Fire {current['fire']}   ·   Smoke {current['smoke']}   ·   Person {current['person']}")
        self.detection_time_var.set(f"Frame time  ·  {result.timestamp.replace('T', ' ')[:19]} UTC")
        if result.detections:
            top_detection = max(result.detections, key=lambda detection: detection.confidence)
            active = self.engine.confirmer.active
            pending = self.engine.confirmer.pending
            if top_detection.class_name in active:
                confirmation = "Confirmed incident · CRITICAL"
            elif top_detection.class_name in pending:
                confirmation = f"Awaiting confirmation · {pending[top_detection.class_name]}/{self.config.confirmation_frames} frames · WARNING"
            elif top_detection.class_name in {"fire", "smoke"} and top_detection.confidence >= self.config.confidence_threshold:
                confirmation = "Detection observed · confirmation streak cleared"
            else:
                confirmation = "Current detection"
            self.detection_detail_var.set(f"{top_detection.class_name.title()}  ·  {top_detection.confidence:.2f} confidence\n{confirmation}")
        else:
            self.detection_detail_var.set("No detections in the latest frame")
        self._set_alert_state(result)

    def _set_alert_state(self, result):
        if result.alert_state == "CRITICAL":
            self.alert_banner.configure(bg="#3a2025"); self.alert_label.configure(bg="#3a2025", fg="#ffc3c4")
            detail = self._active_event_summary or "Confirmed fire / smoke"
            suffix = " · ACKNOWLEDGED" if self._alert_acknowledged else ""
            if self._silenced: suffix += " · ALARM SILENCED"
            self.alert_var.set(f"CONFIRMED INCIDENT  ·  {detail}{suffix}")
            self.ack_button.configure(state="disabled" if self._alert_acknowledged else "normal")
            if self.config.alarm_enabled and not self._silenced and self._last_alert_state != "CRITICAL": self.root.bell()
        elif result.alert_state == "WARNING":
            self.alert_banner.configure(bg="#30291c"); self.alert_label.configure(bg="#30291c", fg=AMBER)
            self.alert_var.set("WARNING  ·  Fire/smoke detection awaiting frame confirmation")
        else:
            self.alert_banner.configure(bg="#17232a"); self.alert_label.configure(bg="#17232a", fg=MUTED)
            self.alert_var.set("MONITORING READY  ·  No confirmed incidents")
            self.ack_button.configure(state="disabled"); self._active_event_id = None; self._silenced = False
            self._active_event_summary = None; self._alert_acknowledged = False
        self._last_alert_state = result.alert_state

    def refresh_events(self):
        rows = self.engine.events.recent_events(None); self._current_events = {r["event_id"]: r for r in rows}
        if hasattr(self, "analytics_total"):
            self.analytics_total.set(str(len(rows)))
            self.analytics_fire.set(str(sum(e.get("event_class", "").lower() == "fire" for e in rows)))
            self.analytics_smoke.set(str(sum(e.get("event_class", "").lower() == "smoke" for e in rows)))
            cameras = sorted({e.get("camera_name", "Camera") for e in rows})
            self.analytics_cameras.set(", ".join(cameras) if cameras else "No recorded events")
            today = datetime.now(timezone.utc).date()
            trend_dates = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
            counts_by_date = {day.isoformat(): 0 for day in trend_dates}
            for event in rows:
                day = event.get("timestamp", "")[:10]
                if day in counts_by_date: counts_by_date[day] += 1
            self.analytics_trend.set("   ·   ".join(f"{day.strftime('%b %d')}  {counts_by_date[day.isoformat()]}" for day in trend_dates))
            self.analytics_health.set(f"{self._camera_name} · {'Connected' if self._connected else 'Disconnected'}")
            self.analytics_recent.set("\n".join(f"{event['timestamp'][11:19]}  ·  {event.get('camera_name', 'Camera')}  ·  {event['event_class'].upper()}  ·  {event['confidence']:.2f}" for event in rows[:5]) if rows else "No recorded events")
        if hasattr(self, "event_camera_box"):
            cameras = ["All cameras", *sorted({e.get("camera_name", "Camera") for e in rows})]
            self.event_camera_box.configure(values=cameras)
            if self.event_camera_filter.get() not in cameras: self.event_camera_filter.set("All cameras")
            severities = ["All severities", *sorted({str(e.get("severity", "Unknown")).title() for e in rows})]
            self.event_severity_box.configure(values=severities)
            if self.event_severity_filter.get() not in severities: self.event_severity_filter.set("All severities")
        if "alerts" in self.page_buttons:
            pending_count = sum(not bool(event.get("acknowledged")) for event in rows)
            self.page_buttons["alerts"].configure(text=f"!   Alerts  ·  {pending_count}" if pending_count else "!   Alerts")
        search = self.event_search.get().strip().lower() if hasattr(self, "event_search") else ""
        event_type = self.event_type_filter.get().lower() if hasattr(self, "event_type_filter") else "all types"
        event_status = self.event_status_filter.get().lower() if hasattr(self, "event_status_filter") else "all statuses"
        event_camera = self.event_camera_filter.get() if hasattr(self, "event_camera_filter") else "All cameras"
        date_after = self.event_after.get().strip() if hasattr(self, "event_after") else ""
        date_before = self.event_before.get().strip() if hasattr(self, "event_before") else ""
        severity_filter = self.event_severity_filter.get().lower() if hasattr(self, "event_severity_filter") else "all severities"
        filtered = [e for e in rows if (not search or search in " ".join(str(e.get(k, "")) for k in ("timestamp", "camera_name", "event_class", "severity")).lower()) and (event_type == "all types" or e.get("event_class", "").lower() == event_type) and (event_status == "all statuses" or bool(e.get("acknowledged")) == (event_status == "acknowledged")) and (event_camera == "All cameras" or e.get("camera_name", "Camera") == event_camera) and (severity_filter == "all severities" or str(e.get("severity", "Unknown")).lower() == severity_filter) and (not date_after or e.get("timestamp", "")[:10] >= date_after) and (not date_before or e.get("timestamp", "")[:10] <= date_before)]
        if hasattr(self, "event_sort") and self.event_sort.get() == "Oldest first": filtered.reverse()
        page_size = 100
        page_count = max(1, (len(filtered) + page_size - 1) // page_size)
        self._event_page = min(getattr(self, "_event_page", 0), page_count - 1)
        if hasattr(self, "event_page_info"):
            start = self._event_page * page_size
            self.event_page_info.set(f"Showing {start + 1 if filtered else 0}–{min(start + page_size, len(filtered))} of {len(filtered)} events  ·  page {self._event_page + 1}/{page_count}")
        for tree in (getattr(self, "event_tree", None), getattr(self, "history_tree", None), getattr(self, "alert_tree", None)):
            if not tree: continue
            for item in tree.get_children(): tree.delete(item)
            visible = rows[:5] if tree is getattr(self, "event_tree", None) else filtered[self._event_page * page_size:(self._event_page + 1) * page_size] if tree is getattr(self, "history_tree", None) else filtered
            for event in visible:
                status = "ACKNOWLEDGED" if event.get("acknowledged") else "UNACKNOWLEDGED"
                values = (event["timestamp"][11:19], event.get("camera_name", "Camera"), event["event_class"].upper(), f'{event["confidence"]:.2f}', event.get("severity", "CRITICAL"), status, "VIEW")
                tree.insert("", "end", iid=event["event_id"], values=values)

    def view_selected_evidence(self, tree):
        selected = tree.selection()
        if not selected: return
        event = self._current_events.get(selected[0]); path = Path(event["evidence_path"]) if event else None
        if path and not path.is_absolute(): path = self.config.evidence_dir / path
        if not path:
            messagebox.showerror("Evidence unavailable", "The saved evidence frame could not be found.", parent=self.root); return
        try:
            resolved_path = path.expanduser().resolve()
            evidence_root = self.config.evidence_dir.expanduser().resolve()
            if not (resolved_path.is_relative_to(evidence_root) or resolved_path.is_relative_to(self.config.project_root.resolve())):
                messagebox.showerror("Evidence unavailable", "This evidence reference is outside the app's evidence locations.", parent=self.root); return
            if not resolved_path.is_file():
                messagebox.showerror("Evidence unavailable", "The saved evidence frame could not be found.", parent=self.root); return
            path = resolved_path
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                messagebox.showerror("Unsupported evidence", f"This desktop viewer cannot preview {path.suffix or 'this file type'}.", parent=self.root); return
            original = Image.open(path).convert("RGB")
            top = tk.Toplevel(self.root); top.title(f'{event.get("camera_name", "Camera")} · {event["event_class"].upper()} · {event["timestamp"]}')
            top.configure(bg=BG); top.geometry("1100x820"); top.minsize(700, 520)
            viewer = tk.Frame(top, bg=BG); viewer.pack(fill="both", expand=True, padx=14, pady=(14, 6))
            canvas = tk.Canvas(viewer, bg="#070c10", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            horizontal = ttk.Scrollbar(viewer, orient="horizontal", command=canvas.xview)
            vertical = ttk.Scrollbar(viewer, orient="vertical", command=canvas.yview)
            canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
            image_item = canvas.create_image(0, 0, anchor="center")
            viewer_state = {"original": False, "photo": None}
            def fit_preview(_event=None):
                if viewer_state["original"]: return
                width = max(320, canvas.winfo_width() - 24); height = max(240, canvas.winfo_height() - 24)
                preview = original.copy(); preview.thumbnail((width, height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(preview); viewer_state["photo"] = photo
                canvas.itemconfigure(image_item, image=photo)
                canvas.coords(image_item, canvas.winfo_width() / 2, canvas.winfo_height() / 2)
                canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), canvas.winfo_height()))
            canvas.bind("<Configure>", fit_preview)
            status = "Acknowledged" if event.get("acknowledged") else "Unacknowledged"
            metadata = f'{event.get("timestamp")}   ·   {event.get("camera_name")}   ·   {event["event_class"].upper()}   ·   {float(event["confidence"]):.2f} confidence   ·   {event.get("severity", "Unknown")}   ·   {status}'
            tk.Label(top, text=metadata, bg=BG, fg=TEXT, font=("Segoe UI", 10), wraplength=1000).pack(pady=(0, 10))
            actions = tk.Frame(top, bg=BG); actions.pack(pady=(0, 12))
            def show_original():
                viewer_state["original"] = True
                viewer_state["photo"] = ImageTk.PhotoImage(original)
                canvas.itemconfigure(image_item, image=viewer_state["photo"])
                canvas.coords(image_item, 0, 0)
                canvas.configure(scrollregion=(0, 0, original.width, original.height))
                vertical.pack(side="right", fill="y"); horizontal.pack(side="bottom", fill="x")
            def fit_window():
                viewer_state["original"] = False
                horizontal.pack_forget(); vertical.pack_forget()
                fit_preview()
            def save_evidence_copy():
                destination = filedialog.asksaveasfilename(parent=top, title="Save evidence copy", initialfile=path.name, defaultextension=path.suffix, filetypes=[("Evidence image", f"*{path.suffix}")])
                if destination:
                    shutil.copy2(path, destination)
                    messagebox.showinfo("Evidence saved", "A copy of the evidence was saved.", parent=top)
            ttk.Button(actions, text="VIEW ORIGINAL", command=show_original).pack(side="left", padx=4)
            ttk.Button(actions, text="FIT TO WINDOW", command=fit_window).pack(side="left", padx=4)
            ttk.Button(actions, text="SAVE COPY", command=save_evidence_copy).pack(side="left", padx=4)
            ttk.Button(actions, text="CLOSE", command=top.destroy).pack(side="left", padx=4)
        except Exception as exc: messagebox.showerror("Evidence error", str(exc), parent=self.root)

    def acknowledge_alert(self):
        selection = self.alert_tree.selection()
        event_id = selection[0] if selection else self._active_event_id
        if event_id:
            self.engine.events.acknowledge(event_id); self._active_event_id = event_id
            self._alert_acknowledged = True
            self.alert_var.set("CRITICAL  ·  Acknowledged — incident remains in event history"); self.ack_button.configure(state="disabled"); self.refresh_events()

    def acknowledge_selected_event(self, tree):
        selection = tree.selection()
        if not selection: return
        self.engine.events.acknowledge(selection[0]); self.refresh_events()

    def silence_alarm(self):
        self._silenced = True; self.alert_var.set("LOCAL ALARM SILENCED  ·  Internal critical events remain active")

    def save_settings(self):
        try:
            confidence = float(self.setting_conf.get()); frames = int(self.setting_frames.get())
            if not 0 <= confidence <= 1 or frames < 1: raise ValueError("Confidence must be 0–1 and confirmation frames must be at least 1.")
            camera = self.setting_camera.get().strip() or "Camera 1"
            rtsp_safe = sanitize_source(self.rtsp_url_var.get().strip())
            new = replace(self.config, model_path=Path(self.setting_model.get()).expanduser().resolve(), confidence_threshold=confidence, confirmation_frames=frames, video_source=self.setting_source.get().strip(), rtsp_url=rtsp_safe, camera_name=camera, evidence_dir=Path(self.setting_evidence.get()).expanduser().resolve(), alarm_enabled=self.alarm_enabled_var.get(), external_alerts_enabled=self.external_enabled_var.get(), email_alerts_enabled=self.email_alert_var.get(), telegram_alerts_enabled=self.telegram_alert_var.get(), webhook_alerts_enabled=self.webhook_alert_var.get())
            save_config(new, rtsp_url=rtsp_safe)
            self.config = new; self.engine.config = new; self.engine.confirmer.threshold = confidence; self.engine.confirmer.confirmation_frames = frames
            self.engine.events.config = new
            self.camera_name_var.set(camera); self.rtsp_url_var.set(rtsp_safe)
            self.health_storage.set(self._storage_summary())
            self.health_logs.set(self._diagnostic_paths_summary())
            messagebox.showinfo("Settings saved", "Settings saved. Detection thresholds apply now; model path applies next launch. External notifications are not implemented.", parent=self.root)
        except Exception as exc: messagebox.showerror("Settings not saved", str(exc), parent=self.root)

    def _poll(self):
        if self._closed: return
        try:
            while True: self._render_frame(self.frames.get_nowait())
        except queue.Empty: pass
        try:
            while True:
                msg = self.messages.get_nowait(); kind = msg[0]
                if kind == "model":
                    self.health_model.set("Loaded"); self._status("System ready", GREEN)
                elif kind == "camera":
                    _, state, error, is_cctv, *details = msg
                    camera_name, source_type = details if len(details) == 2 else (self._camera_name, self._source_type)
                    if state.startswith("RECONNECTING"):
                        try: self._reconnect_attempts = int(state.split("(", 1)[1].split("/", 1)[0])
                        except (ValueError, IndexError): pass
                    self._camera_state(state, error, is_cctv, camera_name, source_type)
                    self.health_camera.set(f"{camera_name} · {state}" + (f" · {error}" if error else ""))
                    if state == "CONNECTED": self._status("Camera connected", GREEN); self.cctv_error.set("")
                    elif state.startswith("RECONNECTING"): self._status(state, AMBER)
                    elif state == "ERROR": self._status("Camera error", "#71383b")
                    elif state == "OFFLINE": self._status("Source offline", PANEL2)
                elif kind == "health": self.health_frames.set(f"Last frame {msg[1][11:19]} UTC · {self._frames_processed:,} frames · {self.fps_value.cget('text')}")
                elif kind == "events":
                    rows = msg[1]; self.refresh_events()
                    if rows:
                        latest = rows[-1]; self._active_event_id = latest["event_id"]
                        self._active_event_summary = f'{latest["event_class"].upper()}  ·  {latest.get("camera_name", self._camera_name)}  ·  {latest["confidence"]:.2f}  ·  {latest["timestamp"]}'
                        self._alert_acknowledged = False
                        self.alert_banner.configure(bg="#3a2025"); self.alert_label.configure(bg="#3a2025", fg="#ffc3c4")
                        self.alert_var.set(f'CONFIRMED INCIDENT  ·  {self._active_event_summary}')
                        self.ack_button.configure(state="normal")
                elif kind == "image_done":
                    self._status("Image processed", GREEN); self.camera_value.configure(text=f"{self._camera_name}  ·  COMPLETE", fg=ACCENT); self.feed_caption.configure(text=f"ANNOTATED RESULT  ·  {Path(msg[1]).name}"); self.refresh_events()
                elif kind == "video_done":
                    self._status("Source stopped", PANEL2); self.refresh_events()
                    self._camera_state("OFFLINE", camera_name=self._camera_name, source_type=self._source_type)
                    if msg[1]: self.feed_caption.configure(text=f"SOURCE ENDED  ·  {msg[1]} frames" + (f"  ·  output {msg[2]}" if msg[2] else ""))
                elif kind == "error":
                    message = msg[1]; is_cctv = msg[2] if len(msg) > 2 else False
                    self._status("Attention required", "#71383b")
                    if message.startswith("Model load failed:"): self.health_model.set(f"Error · {message.removeprefix('Model load failed:').strip()}")
                    if is_cctv: self.cctv_error.set(message)
                    else: messagebox.showerror("FireSight", message, parent=self.root)
                elif kind == "probe_done":
                    self.cctv_error.set(msg[1]); self.reconnect_button.configure(state="disabled")
                elif kind == "idle":
                    self.stop_button.configure(state="disabled")
                    if self.stop_button.winfo_manager(): self.stop_button.pack_forget()
                    if self._pending_action is not None:
                        action, self._pending_action = self._pending_action, None
                        self.root.after(50, action)
        except queue.Empty: pass
        self.root.after(100, self._poll)

    def close(self):
        if self._shutdown_pending or self._closed: return
        self.stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._shutdown_pending = True; self._status("Releasing camera…", AMBER); self.root.after(100, self._finish_shutdown); return
        self._destroy()

    def _finish_shutdown(self):
        if self._worker_thread and self._worker_thread.is_alive(): self.root.after(100, self._finish_shutdown)
        else: self._destroy()

    def _destroy(self): self._closed = True; self.root.destroy()


def launch(config: AppConfig) -> None:
    root = tk.Tk(); FireSightDashboard(root, config); root.mainloop()
