# client_desktop.py - Professional Version
import threading
import time
import cv2
import requests
import sys
import os
from datetime import datetime
from tkinter import Tk, LEFT, RIGHT, BOTH, Label, Button, Frame, StringVar, Canvas, Scrollbar, Text, END, DISABLED, NORMAL
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json

# neuralhash imports - your existing utilities
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash, bits_to_hex
from secretsharing import PlaintextToHexSecretSharer

# --- CONFIG ---
SERVER_URL = "http://127.0.0.1:5000/submit_share"
ACCOUNT_ID = "test_account"
K = 3          # threshold (shares to send)
N = 5          # total shares (not all are sent; we use first K)
CAPTURE_INTERVAL = 3.0   # seconds between automatic captures
CAMERA_INDEX = 0         # default webcam

MODEL_PCA_PATH = "../models/pca_512_to_128.pkl"
MODEL_HYPER_PATH = "../models/neuralhash_128x96_seed1.dat"

BUZZER_WAV = "buzzer.wav" 
# ------------------

# load models (this may take time)
print("[INFO] Loading PCA model and hyperplanes...")
pca = load_pca_model(MODEL_PCA_PATH)
hyperplanes = load_hyperplanes(MODEL_HYPER_PATH)
print("[INFO] Models loaded.")

# Try to set up audio playback (simpleaudio preferred)
_audio_player = None
try:
    import simpleaudio as sa
    _audio_player = "simpleaudio"
    if os.path.exists(BUZZER_WAV):
        buzzer_wave = sa.WaveObject.from_wave_file(BUZZER_WAV)
    else:
        buzzer_wave = None
except Exception:
    buzzer_wave = None
    _audio_player = None

# Windows fallback
if _audio_player is None:
    try:
        import winsound
        _audio_player = "winsound"
    except Exception:
        _audio_player = None

class ProfessionalColors:
    # Dark professional theme
    BACKGROUND = "#1a1a1a"
    SIDEBAR = "#2d2d2d"
    ACCENT = "#0078d4"
    SUCCESS = "#107c10"
    WARNING = "#ff8c00"
    DANGER = "#d13438"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_MUTED = "#8a8a8a"
    BORDER = "#404040"
    HOVER = "#3d3d3d"

# GUI Application
class ProfessionalBorderControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Border Control Face Match")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        self.root.configure(bg=ProfessionalColors.BACKGROUND)
        
        # Set window icon and styling
        self.setup_styling()
        
        # Initialize data tracking
        self.scan_count = 0
        self.match_count = 0
        self.last_scan_time = None
        self.activity_log = []
        self.system_status = "ONLINE"
        
        # Create main layout
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
        # Initialize camera and threading
        self.initialize_system()
        
        # Start status updates
        self.update_system_time()

    def setup_styling(self):
        """Configure ttk styling for professional look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Professional.TButton',
                       background=ProfessionalColors.ACCENT,
                       foreground=ProfessionalColors.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10))
        
        style.configure('Danger.TButton',
                       background=ProfessionalColors.DANGER,
                       foreground=ProfessionalColors.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'))

    def create_header(self):
        """Create professional header with branding and system info"""
        self.header_frame = Frame(self.root, bg=ProfessionalColors.SIDEBAR, height=80)
        self.header_frame.pack(fill="x", padx=0, pady=0)
        self.header_frame.pack_propagate(False)
        
        # Left side - Logo and title
        left_header = Frame(self.header_frame, bg=ProfessionalColors.SIDEBAR)
        left_header.pack(side=LEFT, fill="y", padx=20, pady=15)
        
        # Logo placeholder (you can replace with actual logo)
        self.logo_canvas = Canvas(left_header, width=50, height=50, bg=ProfessionalColors.ACCENT, highlightthickness=0)
        self.logo_canvas.pack(side=LEFT, padx=(0,15))
        self.logo_canvas.create_text(25, 25, text="NH", fill="white", font=('Segoe UI', 12, 'bold'))
        
        title_frame = Frame(left_header, bg=ProfessionalColors.SIDEBAR)
        title_frame.pack(side=LEFT)
        
        Label(title_frame, text="Border Control Face Matcher", 
              font=('Segoe UI', 16, 'bold'), fg=ProfessionalColors.TEXT_PRIMARY, 
              bg=ProfessionalColors.SIDEBAR).pack(anchor="w")
        Label(title_frame, text="NeuralHash Match Verification Platform", 
              font=('Segoe UI', 9), fg=ProfessionalColors.TEXT_MUTED, 
              bg=ProfessionalColors.SIDEBAR).pack(anchor="w")
        
        # Right side - System status and time
        right_header = Frame(self.header_frame, bg=ProfessionalColors.SIDEBAR)
        right_header.pack(side=RIGHT, fill="y", padx=20, pady=15)
        
        self.time_label = Label(right_header, text="", font=('Segoe UI', 11), 
                               fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR)
        self.time_label.pack(anchor="e")
        
        status_frame = Frame(right_header, bg=ProfessionalColors.SIDEBAR)
        status_frame.pack(anchor="e")
        
        self.status_indicator = Canvas(status_frame, width=12, height=12, bg=ProfessionalColors.SIDEBAR, highlightthickness=0)
        self.status_indicator.pack(side=LEFT, padx=(0,5))
        self.status_indicator.create_oval(2, 2, 10, 10, fill=ProfessionalColors.SUCCESS, outline="")
        
        Label(status_frame, text="SYSTEM ONLINE", font=('Segoe UI', 9, 'bold'), 
              fg=ProfessionalColors.SUCCESS, bg=ProfessionalColors.SIDEBAR).pack(side=LEFT)

    def create_main_content(self):
        """Create main content area with video feed and control panels"""
        self.main_frame = Frame(self.root, bg=ProfessionalColors.BACKGROUND)
        self.main_frame.pack(fill=BOTH, expand=True, padx=10, pady=(0,10))
        
        # Left panel - Video feed
        self.create_video_panel()
        
        # Right panel - Controls and information
        self.create_control_panel()

    def create_video_panel(self):
        """Create professional video feed panel"""
        self.video_panel = Frame(self.main_frame, bg=ProfessionalColors.SIDEBAR, relief="solid", bd=1)
        self.video_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0,10))
        
        # Video header
        video_header = Frame(self.video_panel, bg=ProfessionalColors.BORDER, height=40)
        video_header.pack(fill="x")
        video_header.pack_propagate(False)
        
        Label(video_header, text="🎥 LIVE CAMERA FEED", font=('Segoe UI', 11, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BORDER).pack(side=LEFT, padx=15, pady=10)
        
        # Camera status
        self.camera_status = Label(video_header, text="● ACTIVE", font=('Segoe UI', 9, 'bold'),
                                  fg=ProfessionalColors.SUCCESS, bg=ProfessionalColors.BORDER)
        self.camera_status.pack(side=RIGHT, padx=15, pady=10)
        
        # Video display area
        self.video_container = Frame(self.video_panel, bg=ProfessionalColors.BACKGROUND)
        self.video_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        self.video_label = Label(self.video_container, bg=ProfessionalColors.BACKGROUND, 
                                text="Initializing camera...", fg=ProfessionalColors.TEXT_MUTED)
        self.video_label.pack(expand=True)
        
        # Video overlay for scan indicator
        self.create_scan_overlay()

    def create_scan_overlay(self):
        """Create overlay graphics for scanning indication"""
        # Note: Overlay graphics are now handled directly in the video processing
        # This method is kept for future overlay enhancements
        pass
        
    def create_control_panel(self):
        """Create comprehensive control and monitoring panel"""
        self.control_panel = Frame(self.main_frame, bg=ProfessionalColors.SIDEBAR, width=400, relief="solid", bd=1)
        self.control_panel.pack(side=RIGHT, fill="y")
        self.control_panel.pack_propagate(False)
        
        # Create tabbed interface
        self.create_tabbed_controls()

    def create_tabbed_controls(self):
        """Create tabbed control interface"""
        # Tab buttons
        tab_frame = Frame(self.control_panel, bg=ProfessionalColors.BORDER, height=50)
        tab_frame.pack(fill="x")
        tab_frame.pack_propagate(False)
        
        self.active_tab = "status"
        self.tabs = {
            "status": {"name": "Status", "button": None},
            "activity": {"name": "Activity", "button": None},
            "settings": {"name": "Settings", "button": None}
        }
        
        for i, (tab_id, tab_data) in enumerate(self.tabs.items()):
            btn = Button(tab_frame, text=tab_data["name"], 
                        command=lambda t=tab_id: self.switch_tab(t),
                        bg=ProfessionalColors.ACCENT if tab_id == "status" else ProfessionalColors.BORDER,
                        fg=ProfessionalColors.TEXT_PRIMARY,
                        font=('Segoe UI', 10), relief="flat", bd=0)
            btn.pack(side=LEFT, fill="x", expand=True)
            tab_data["button"] = btn
        
        # Tab content area
        self.tab_content = Frame(self.control_panel, bg=ProfessionalColors.SIDEBAR)
        self.tab_content.pack(fill=BOTH, expand=True)
        
        # Create all tab contents
        self.create_status_tab()
        self.create_activity_tab()
        self.create_settings_tab()
        
        # Show default tab
        self.switch_tab("status")

    def create_status_tab(self):
        """Create status monitoring tab"""
        self.status_tab = Frame(self.tab_content, bg=ProfessionalColors.SIDEBAR)
        
        # Alert Section
        alert_section = Frame(self.status_tab, bg=ProfessionalColors.SIDEBAR)
        alert_section.pack(fill="x", padx=20, pady=20)
        
        Label(alert_section, text="THREAT DETECTION", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w")
        
        # Alert status box
        self.alert_frame = Frame(alert_section, bg=ProfessionalColors.BACKGROUND, relief="solid", bd=2, height=120)
        self.alert_frame.pack(fill="x", pady=(10,0))
        self.alert_frame.pack_propagate(False)
        
        self.alert_icon = Label(self.alert_frame, text="✅", font=('Segoe UI', 32),
                               fg=ProfessionalColors.SUCCESS, bg=ProfessionalColors.BACKGROUND)
        self.alert_icon.pack(pady=10)
        
        self.alert_status = Label(self.alert_frame, text="NO THREATS DETECTED", 
                                 font=('Segoe UI', 14, 'bold'),
                                 fg=ProfessionalColors.SUCCESS, bg=ProfessionalColors.BACKGROUND)
        self.alert_status.pack()
        
        self.alert_details = Label(self.alert_frame, text="System monitoring normally", 
                                  font=('Segoe UI', 10),
                                  fg=ProfessionalColors.TEXT_MUTED, bg=ProfessionalColors.BACKGROUND)
        self.alert_details.pack()
        
        # Clear alarm button
        self.clear_alarm_btn = ttk.Button(alert_section, text="🔕 CLEAR ALERT", 
                                         style='Danger.TButton',
                                         command=self.clear_alarm, state="disabled")
        self.clear_alarm_btn.pack(pady=(15,0))
        
        # Statistics Section
        stats_section = Frame(self.status_tab, bg=ProfessionalColors.SIDEBAR)
        stats_section.pack(fill="x", padx=20, pady=(0,20))
        
        Label(stats_section, text="SESSION STATISTICS", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w")
        
        stats_grid = Frame(stats_section, bg=ProfessionalColors.SIDEBAR)
        stats_grid.pack(fill="x", pady=(10,0))
        
        # Create stat cards
        self.create_stat_card(stats_grid, "Total Scans", "0", 0, 0)
        self.create_stat_card(stats_grid, "Matches Found", "0", 0, 1)
        self.create_stat_card(stats_grid, "Last Scan", "Never", 1, 0)
        self.create_stat_card(stats_grid, "System Uptime", "00:00:00", 1, 1)
        
        # Control buttons
        control_section = Frame(self.status_tab, bg=ProfessionalColors.SIDEBAR)
        control_section.pack(fill="x", padx=20, pady=(0,20))
        
        Label(control_section, text="MANUAL CONTROLS", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w")
        
        self.scan_btn = ttk.Button(control_section, text="🔍 SCAN NOW", 
                                  style='Professional.TButton',
                                  command=self.capture_now)
        self.scan_btn.pack(fill="x", pady=(10,5))
        
        # Auto-scan toggle
        auto_frame = Frame(control_section, bg=ProfessionalColors.SIDEBAR)
        auto_frame.pack(fill="x", pady=5)
        
        self.auto_scan_var = StringVar(value=f"🔄 Auto-scan: {CAPTURE_INTERVAL}s")
        self.auto_scan_label = Label(auto_frame, textvariable=self.auto_scan_var,
                                    font=('Segoe UI', 10), fg=ProfessionalColors.TEXT_MUTED,
                                    bg=ProfessionalColors.SIDEBAR)
        self.auto_scan_label.pack()

    def create_stat_card(self, parent, title, value, row, col):
        """Create a statistics card"""
        card = Frame(parent, bg=ProfessionalColors.BACKGROUND, relief="solid", bd=1)
        card.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        
        Label(card, text=title, font=('Segoe UI', 9),
              fg=ProfessionalColors.TEXT_MUTED, bg=ProfessionalColors.BACKGROUND).pack(pady=(8,2))
        
        value_label = Label(card, text=value, font=('Segoe UI', 14, 'bold'),
                           fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND)
        value_label.pack(pady=(0,8))
        
        # Store reference for updates
        if title == "Total Scans":
            self.total_scans_label = value_label
        elif title == "Matches Found":
            self.matches_found_label = value_label
        elif title == "Last Scan":
            self.last_scan_label = value_label
        elif title == "System Uptime":
            self.uptime_label = value_label

    def create_activity_tab(self):
        """Create activity log tab"""
        self.activity_tab = Frame(self.tab_content, bg=ProfessionalColors.SIDEBAR)
        
        # Header
        header = Frame(self.activity_tab, bg=ProfessionalColors.SIDEBAR)
        header.pack(fill="x", padx=20, pady=(20,10))
        
        Label(header, text="ACTIVITY LOG", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(side=LEFT)
        
        clear_log_btn = Button(header, text="Clear Log", command=self.clear_activity_log,
                              bg=ProfessionalColors.BORDER, fg=ProfessionalColors.TEXT_PRIMARY,
                              font=('Segoe UI', 9), relief="flat", bd=0)
        clear_log_btn.pack(side=RIGHT)
        
        # Activity log display
        log_frame = Frame(self.activity_tab, bg=ProfessionalColors.SIDEBAR)
        log_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0,20))
        
        # Scrollable text widget
        self.activity_text = Text(log_frame, bg=ProfessionalColors.BACKGROUND,
                                 fg=ProfessionalColors.TEXT_PRIMARY, font=('Consolas', 9),
                                 wrap="word", state=DISABLED, relief="solid", bd=1)
        
        scrollbar = Scrollbar(log_frame)
        scrollbar.pack(side=RIGHT, fill="y")
        self.activity_text.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar.config(command=self.activity_text.yview)
        self.activity_text.config(yscrollcommand=scrollbar.set)
        
        # Add initial log entry
        self.log_activity("SYSTEM", "Border Control System initialized", "INFO")

    def create_settings_tab(self):
        """Create settings configuration tab"""
        self.settings_tab = Frame(self.tab_content, bg=ProfessionalColors.SIDEBAR)
        
        # Settings content
        settings_scroll = Frame(self.settings_tab, bg=ProfessionalColors.SIDEBAR)
        settings_scroll.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Camera Settings
        Label(settings_scroll, text="CAMERA SETTINGS", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w", pady=(0,10))
        
        cam_frame = Frame(settings_scroll, bg=ProfessionalColors.BACKGROUND, relief="solid", bd=1)
        cam_frame.pack(fill="x", pady=(0,15))
        
        Label(cam_frame, text=f"Camera Index: {CAMERA_INDEX}", font=('Segoe UI', 10),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND).pack(pady=10, padx=15, anchor="w")
        
        Label(cam_frame, text=f"Capture Interval: {CAPTURE_INTERVAL}s", font=('Segoe UI', 10),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND).pack(pady=(0,10), padx=15, anchor="w")
        
        # System Settings
        Label(settings_scroll, text="SYSTEM SETTINGS", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w", pady=(15,10))
        
        sys_frame = Frame(settings_scroll, bg=ProfessionalColors.BACKGROUND, relief="solid", bd=1)
        sys_frame.pack(fill="x", pady=(0,15))
        
        Label(sys_frame, text=f"Server: {SERVER_URL}", font=('Segoe UI', 10),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND).pack(pady=10, padx=15, anchor="w")
        
        Label(sys_frame, text=f"Account: {ACCOUNT_ID}", font=('Segoe UI', 10),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND).pack(pady=(0,10), padx=15, anchor="w")
        
        # Model Information
        Label(settings_scroll, text="MODEL INFORMATION", font=('Segoe UI', 12, 'bold'),
              fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.SIDEBAR).pack(anchor="w", pady=(15,10))
        
        model_frame = Frame(settings_scroll, bg=ProfessionalColors.BACKGROUND, relief="solid", bd=1)
        model_frame.pack(fill="x")
        
        model_info = [
            f"PCA Model: {os.path.basename(MODEL_PCA_PATH)}",
            f"Hyperplanes: {os.path.basename(MODEL_HYPER_PATH)}",
            f"Secret Sharing: {K}/{N} threshold"
        ]
        
        for info in model_info:
            Label(model_frame, text=info, font=('Segoe UI', 10),
                  fg=ProfessionalColors.TEXT_PRIMARY, bg=ProfessionalColors.BACKGROUND).pack(pady=5, padx=15, anchor="w")

    def create_footer(self):
        """Create professional footer"""
        self.footer_frame = Frame(self.root, bg=ProfessionalColors.BORDER, height=30)
        self.footer_frame.pack(fill="x", side="bottom")
        self.footer_frame.pack_propagate(False)
        
        Label(self.footer_frame, text="NeuralHash Border Control System © 2024 | Authorized Personnel Only", 
              font=('Segoe UI', 8), fg=ProfessionalColors.TEXT_MUTED, 
              bg=ProfessionalColors.BORDER).pack(pady=8)

    def switch_tab(self, tab_id):
        """Switch between control panel tabs"""
        # Hide all tabs
        for widget in self.tab_content.winfo_children():
            widget.pack_forget()
        
        # Update button colors
        for tid, tab_data in self.tabs.items():
            color = ProfessionalColors.ACCENT if tid == tab_id else ProfessionalColors.BORDER
            tab_data["button"].configure(bg=color)
        
        # Show selected tab
        if tab_id == "status":
            self.status_tab.pack(fill=BOTH, expand=True)
        elif tab_id == "activity":
            self.activity_tab.pack(fill=BOTH, expand=True)
        elif tab_id == "settings":
            self.settings_tab.pack(fill=BOTH, expand=True)
        
        self.active_tab = tab_id

    def initialize_system(self):
        """Initialize camera and threading systems"""
        self.vs = cv2.VideoCapture(CAMERA_INDEX)
        if not self.vs.isOpened():
            self.log_activity("ERROR", "Cannot open webcam. System offline.", "ERROR")
            self.camera_status.configure(text="● ERROR", fg=ProfessionalColors.DANGER)
            return

        self.running = True
        self.alarm_playing = False
        self.alarm_stop_event = threading.Event()
        self.is_match_active = False
        self.start_time = time.time()

        # Start threads
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.update_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.update_thread.start()

        self.capture_thread = threading.Thread(target=self._periodic_capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.log_activity("SYSTEM", "Camera initialized successfully", "INFO")

    def update_system_time(self):
        """Update system time and uptime"""
        if self.running:
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.configure(text=time_str)
            
            # Update uptime
            if hasattr(self, 'start_time'):
                uptime = time.time() - self.start_time
                hours = int(uptime // 3600)
                minutes = int((uptime % 3600) // 60)
                seconds = int(uptime % 60)
                self.uptime_label.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            self.root.after(1000, self.update_system_time)

    def log_activity(self, source, message, level="INFO"):
        """Add entry to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": ProfessionalColors.TEXT_PRIMARY,
            "WARN": ProfessionalColors.WARNING,
            "ERROR": ProfessionalColors.DANGER,
            "SUCCESS": ProfessionalColors.SUCCESS
        }
        
        self.activity_text.configure(state=NORMAL)
        self.activity_text.insert(END, f"[{timestamp}] [{source}] {message}\n")
        self.activity_text.configure(state=DISABLED)
        self.activity_text.see(END)
        
        # Keep log size manageable
        if len(self.activity_log) > 100:
            self.activity_log.pop(0)
        self.activity_log.append(f"[{timestamp}] [{source}] {message}")

    def clear_activity_log(self):
        """Clear the activity log"""
        self.activity_text.configure(state=NORMAL)
        self.activity_text.delete(1.0, END)
        self.activity_text.configure(state=DISABLED)
        self.activity_log.clear()
        self.log_activity("SYSTEM", "Activity log cleared", "INFO")

    def _video_loop(self):
        """Enhanced video loop with professional overlay"""
        while self.running:
            ret, frame = self.vs.read()
            if not ret:
                continue
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Add professional overlay
            frame_with_overlay = self.add_video_overlay(frame)
            
            # Convert for display
            rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # Resize to fit display
            display_size = (800, 600)
            img = img.resize(display_size, Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            def update_video():
                if self.running:
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk, text="")
            
            self.root.after_idle(update_video)
            time.sleep(0.033)  # ~30 FPS

    def add_video_overlay(self, frame):
        """Add professional overlay graphics to video"""
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Add corner brackets for scanning area
        bracket_size = 50
        thickness = 3
        color = (0, 120, 212) if not self.is_match_active else (209, 52, 56)
        
        # Top-left
        cv2.line(overlay_frame, (50, 50), (50 + bracket_size, 50), color, thickness)
        cv2.line(overlay_frame, (50, 50), (50, 50 + bracket_size), color, thickness)
        
        # Top-right
        cv2.line(overlay_frame, (width - 50, 50), (width - 50 - bracket_size, 50), color, thickness)
        cv2.line(overlay_frame, (width - 50, 50), (width - 50, 50 + bracket_size), color, thickness)
        
        # Bottom-left
        cv2.line(overlay_frame, (50, height - 50), (50 + bracket_size, height - 50), color, thickness)
        cv2.line(overlay_frame, (50, height - 50), (50, height - 50 - bracket_size), color, thickness)
        
        # Bottom-right
        cv2.line(overlay_frame, (width - 50, height - 50), (width - 50 - bracket_size, height - 50), color, thickness)
        cv2.line(overlay_frame, (width - 50, height - 50), (width - 50, height - 50 - bracket_size), color, thickness)
        
        # Add center crosshair
        center_x, center_y = width // 2, height // 2
        cv2.line(overlay_frame, (center_x - 20, center_y), (center_x + 20, center_y), color, 2)
        cv2.line(overlay_frame, (center_x, center_y - 20), (center_x, center_y + 20), color, 2)
        
        # Add status text overlay
        status_text = "SCANNING..." if self.is_match_active else "MONITORING"
        cv2.putText(overlay_frame, status_text, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return overlay_frame

    def _periodic_capture_loop(self):
        """Enhanced periodic capture with better logging"""
        while self.running:
            if not self.is_match_active:
                self._capture_process_and_send()
            time.sleep(CAPTURE_INTERVAL)

    def capture_now(self):
        """Manual capture with UI feedback"""
        self.log_activity("USER", "Manual scan initiated", "INFO")
        threading.Thread(target=self._capture_process_and_send, daemon=True).start()

    def _capture_process_and_send(self):
        """Enhanced capture processing with detailed logging"""
        # Acquire latest frame
        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()
        if frame is None:
            self.log_activity("ERROR", "No camera frame available", "ERROR")
            return

        # Update scan count
        self.scan_count += 1
        self.total_scans_label.configure(text=str(self.scan_count))
        self.last_scan_time = datetime.now()
        self.last_scan_label.configure(text=self.last_scan_time.strftime("%H:%M:%S"))

        # Save temp image for hashing utils
        tmp_path = "../temp_captures/temp_capture.jpg"
        cv2.imwrite(tmp_path, frame)
        
        self.log_activity("SCAN", f"Capture #{self.scan_count} processed", "INFO")

        try:
            # Generate hash
            bits = generate_neuralhash(tmp_path, pca, hyperplanes)
            hex_hash = bits_to_hex(bits)
            self.log_activity("HASH", f"Neural hash generated: {hex_hash[:16]}...", "INFO")

            # Split into shares
            shares = PlaintextToHexSecretSharer.split_secret(hex_hash, K, N)
            self.log_activity("CRYPTO", f"Secret split into {len(shares)} shares", "INFO")

            # Send shares to server
            payload_result = None
            for i, share in enumerate(shares[:K], start=1):
                payload = {'account': ACCOUNT_ID, 'share': share, 'original_hex': hex_hash}
                try:
                    r = requests.post(SERVER_URL, json=payload, timeout=6, proxies={})
                    payload_result = r.json()
                    self.log_activity("NET", f"Share {i}/{K} sent successfully", "SUCCESS")
                except Exception as e:
                    self.log_activity("ERROR", f"Failed to send share {i}: {str(e)}", "ERROR")
                    payload_result = None
                    break

            # Process server response
            if payload_result and isinstance(payload_result, dict):
                status = payload_result.get("status")
                if status == "match":
                    matches = payload_result.get("matches", [])
                    if matches:
                        self.match_count += 1
                        self.matches_found_label.configure(text=str(self.match_count))
                        
                        first = matches[0]
                        matched_id = first.get("id") or first.get("ref_id") or ""
                        display_name = self._extract_person_name(matched_id)
                        distance = first.get("distance")
                        
                        self.log_activity("ALERT", f"MATCH FOUND: {display_name} (distance: {distance})", "WARN")
                        self._trigger_match_ui(display_name, distance, matched_id)
                else:
                    # No match
                    if self.is_match_active:
                        self._clear_match_ui()
                    self.log_activity("SCAN", "No matches detected", "INFO")
            else:
                self.log_activity("ERROR", "Invalid server response", "ERROR")

        except Exception as e:
            self.log_activity("ERROR", f"Processing failed: {str(e)}", "ERROR")

    def _extract_person_name(self, ref_id):
        """Extract person name from reference ID"""
        if not ref_id:
            return "Unknown"
        name = ref_id.split("_")[0]
        name = name.split(".")[0]
        return name.title()

    def _trigger_match_ui(self, person_name, distance=None, ref_id=None):
        """Enhanced match UI with professional styling"""
        def gui_update():
            self.is_match_active = True
            
            # Update alert display
            self.alert_icon.configure(text="🚨", fg=ProfessionalColors.DANGER)
            self.alert_status.configure(text=f"THREAT DETECTED", fg=ProfessionalColors.DANGER)
            
            distance_text = f" (Confidence: {100-int(distance) if distance else 95}%)" if distance else ""
            self.alert_details.configure(text=f"Individual: {person_name}{distance_text}")
            
            # Style the alert frame
            self.alert_frame.configure(bg=ProfessionalColors.DANGER, bd=3)
            self.alert_icon.configure(bg=ProfessionalColors.DANGER)
            self.alert_status.configure(bg=ProfessionalColors.DANGER)
            self.alert_details.configure(bg=ProfessionalColors.DANGER, fg=ProfessionalColors.TEXT_PRIMARY)
            
            # Enable clear button
            self.clear_alarm_btn.configure(state="normal")
            
            # Start visual effects
            self._start_enhanced_flashing()
            self._start_buzzer()
            
            # Update system status
            self.status_indicator.create_oval(2, 2, 10, 10, fill=ProfessionalColors.DANGER, outline="")
            
        self.root.after(0, gui_update)

    def _clear_match_ui(self):
        """Enhanced clear match UI"""
        def gui_update():
            self.is_match_active = False
            
            # Reset alert display
            self.alert_icon.configure(text="✅", fg=ProfessionalColors.SUCCESS)
            self.alert_status.configure(text="NO THREATS DETECTED", fg=ProfessionalColors.SUCCESS)
            self.alert_details.configure(text="System monitoring normally")
            
            # Reset alert frame styling
            self.alert_frame.configure(bg=ProfessionalColors.BACKGROUND, bd=2)
            self.alert_icon.configure(bg=ProfessionalColors.BACKGROUND)
            self.alert_status.configure(bg=ProfessionalColors.BACKGROUND)
            self.alert_details.configure(bg=ProfessionalColors.BACKGROUND, fg=ProfessionalColors.TEXT_MUTED)
            
            # Disable clear button
            self.clear_alarm_btn.configure(state="disabled")
            
            # Stop effects
            self._stop_flashing()
            self._stop_buzzer()
            
            # Reset system status
            self.status_indicator.create_oval(2, 2, 10, 10, fill=ProfessionalColors.SUCCESS, outline="")
            
        self.root.after(0, gui_update)

    def _start_enhanced_flashing(self):
        """Enhanced flashing with multiple elements"""
        self._flash_on = True
        self._flash_state = True
        
        def flash():
            if getattr(self, "_flash_on", False) and self.is_match_active:
                # Alternate between two danger colors
                bg_color = ProfessionalColors.DANGER if self._flash_state else "#ff6b6b"
                
                self.alert_frame.configure(bg=bg_color)
                self.alert_icon.configure(bg=bg_color)
                self.alert_status.configure(bg=bg_color)
                self.alert_details.configure(bg=bg_color)
                
                # Toggle state
                self._flash_state = not self._flash_state
                
                # Continue flashing
                self.root.after(400, flash)
                
        self.root.after(0, flash)

    def _stop_flashing(self):
        """Stop flashing effect"""
        self._flash_on = False

    def _start_buzzer(self):
        """Enhanced buzzer system"""
        if self.alarm_playing:
            return
        self.alarm_stop_event.clear()
        self.alarm_playing = True

        def buzzer_loop():
            if _audio_player == "simpleaudio" and buzzer_wave is not None:
                while not self.alarm_stop_event.is_set():
                    play_obj = buzzer_wave.play()
                    while play_obj.is_playing() and not self.alarm_stop_event.is_set():
                        time.sleep(0.1)
                    if not self.alarm_stop_event.is_set():
                        time.sleep(0.3)
            elif _audio_player == "winsound":
                try:
                    winsound.PlaySound(BUZZER_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                    while not self.alarm_stop_event.is_set():
                        time.sleep(0.2)
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception as e:
                    self.log_activity("ERROR", f"Audio system error: {str(e)}", "ERROR")
            else:
                # Visual-only fallback
                while not self.alarm_stop_event.is_set():
                    time.sleep(0.5)
            
            self.alarm_playing = False

        threading.Thread(target=buzzer_loop, daemon=True).start()

    def _stop_buzzer(self):
        """Stop buzzer system"""
        if not self.alarm_playing:
            return
        self.alarm_stop_event.set()
        time.sleep(0.1)
        self.alarm_playing = False

    def clear_alarm(self):
        """Clear alarm with logging"""
        self.log_activity("USER", "Security alert cleared by operator", "INFO")
        self._clear_match_ui()

    def shutdown(self):
        """Enhanced shutdown procedure"""
        self.log_activity("SYSTEM", "Initiating system shutdown...", "INFO")
        self.running = False
        self._stop_buzzer()
        time.sleep(0.3)
        
        try:
            self.vs.release()
        except:
            pass
            
        self.log_activity("SYSTEM", "Border Control System offline", "INFO")
        self.root.quit()


if __name__ == "__main__":
    root = Tk()
    
    # Set application icon (if available)
    try:
        root.iconbitmap("icon.ico")  # Add your icon file
    except:
        pass
    
    app = ProfessionalBorderControlApp(root)
    try:
        root.protocol("WM_DELETE_WINDOW", app.shutdown)
        root.mainloop()
    except KeyboardInterrupt:
        app.shutdown()