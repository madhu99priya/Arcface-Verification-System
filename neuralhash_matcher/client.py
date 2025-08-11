# client_desktop.py
import threading
import time
import cv2
import requests
import sys
import os
from tkinter import Tk, LEFT, RIGHT, BOTH, Label, Button, Frame, StringVar
from PIL import Image, ImageTk

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

MODEL_PCA_PATH = "./models/pca_512_to_128.pkl"
MODEL_HYPER_PATH = "./models/neuralhash_128x96_seed1.dat"

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

# GUI Application
class NeuralHashClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Border Control — Face Match (Demo)")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        # Frames
        self.left_frame = Frame(root, width=680, height=600, bg="#222")
        self.left_frame.pack(side=LEFT, fill=BOTH)
        self.right_frame = Frame(root, width=320, height=600, bg="#111")
        self.right_frame.pack(side=RIGHT, fill=BOTH)

        # Camera label (for feed)
        self.video_label = Label(self.left_frame)
        self.video_label.pack(expand=True)

        # Status area
        self.status_title = StringVar()
        self.status_title.set("Status")
        Label(self.right_frame, textvariable=self.status_title, font=("Helvetica", 14), fg="#ddd", bg="#111").pack(pady=(24,8))

        self.match_label = Label(self.right_frame, text="No Match Detected", font=("Helvetica", 20, "bold"),
                                 fg="#00c853", bg="#111", wraplength=280, justify="center")
        self.match_label.pack(pady=(16,8), ipadx=8, ipady=8)

        self.person_label = Label(self.right_frame, text="", font=("Helvetica", 16), fg="#fff", bg="#111")
        self.person_label.pack(pady=(8,8))

        # Clear alarm button (hidden until match)
        self.clear_btn = Button(self.right_frame, text="Clear Alarm", command=self.clear_alarm, state="disabled",
                                bg="#444", fg="#fff", padx=12, pady=8)
        self.clear_btn.pack(pady=(20,8))

        # Capture control
        self.capture_btn = Button(self.right_frame, text="Capture Now", command=self.capture_now,
                                  bg="#0066cc", fg="#fff", padx=12, pady=8)
        self.capture_btn.pack(pady=(10,4))

        # Config
        self.interval_label = Label(self.right_frame, text=f"Auto-capture every {CAPTURE_INTERVAL:.1f}s", fg="#aaa", bg="#111")
        self.interval_label.pack(pady=(6,4))

        # State
        self.vs = cv2.VideoCapture(CAMERA_INDEX)
        if not self.vs.isOpened():
            print("[ERROR] Cannot open webcam. Exiting.")
            sys.exit(1)

        self.running = True
        self.alarm_playing = False
        self.alarm_stop_event = threading.Event()

        # Start threads
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.update_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.update_thread.start()

        self.capture_thread = threading.Thread(target=self._periodic_capture_loop, daemon=True)
        self.capture_thread.start()

    def _video_loop(self):
        while self.running:
            ret, frame = self.vs.read()
            if not ret:
                continue
            with self.frame_lock:
                self.current_frame = frame.copy()
            # Convert for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((680, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            # update label in main thread
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            time.sleep(0.02)

    def _periodic_capture_loop(self):
        while self.running:
            self._capture_process_and_send()
            time.sleep(CAPTURE_INTERVAL)

    def capture_now(self):
        # manual capture
        threading.Thread(target=self._capture_process_and_send, daemon=True).start()

    def _capture_process_and_send(self):
        # Acquire latest frame
        with self.frame_lock:
            frame = None if self.current_frame is None else self.current_frame.copy()
        if frame is None:
            print("[WARN] No frame available to capture.")
            return

        # Save temp image for hashing utils
        tmp_path = "temp_capture.jpg"
        cv2.imwrite(tmp_path, frame)
        print("[INFO] Captured image ->", tmp_path)

        try:
            bits = generate_neuralhash(tmp_path, pca, hyperplanes)
            hex_hash = bits_to_hex(bits)
            print(f"[INFO] NeuralHash (hex): {hex_hash}")

            shares = PlaintextToHexSecretSharer.split_secret(hex_hash, K, N)
            print(f"[INFO] Split into {len(shares)} shares. Sending first {K} shares...")

            # send first K shares with original_hex to server
            payload_result = None
            for i, share in enumerate(shares[:K], start=1):
                payload = {'account': ACCOUNT_ID, 'share': share, 'original_hex': hex_hash}
                try:
                    # disable proxies to avoid environment proxy issues
                    r = requests.post(SERVER_URL, json=payload, timeout=6, proxies={})
                    payload_result = r.json()
                    print(f"[INFO] Sent share {i}/{K}. Server response: {payload_result}")
                except Exception as e:
                    print("[ERROR] Failed to send share:", e)
                    payload_result = None
                    break

            # If we got a response and it's a match, trigger UI
            if payload_result and isinstance(payload_result, dict):
                status = payload_result.get("status")
                if status == "match":
                    matches = payload_result.get("matches", [])
                    # pick the best/first match
                    if matches:
                        first = matches[0]
                        matched_id = first.get("id") or first.get("ref_id") or ""
                        # derive folder/name display - e.g. "Madhusha_frame_4.jpg" -> "Madhusha"
                        display_name = self._extract_person_name(matched_id)
                        distance = first.get("distance")
                        self._trigger_match_ui(display_name, distance)
                else:
                    # no match
                    self._clear_match_ui()

        except Exception as e:
            print("[ERROR] Hash generation or splitting failed:", e)

    def _extract_person_name(self, ref_id):
        if not ref_id:
            return "Unknown"
        # try split at underscore first, else dot
        name = ref_id.split("_")[0]
        name = name.split(".")[0]
        return name

    def _trigger_match_ui(self, person_name, distance=None):
        # update UI on main thread
        def gui_update():
            self.match_label.configure(text=f"🚨 MATCH: {person_name}", fg="#ff1744", bg="#880000")
            self.person_label.configure(text=f"{person_name}" + (f"  (dist: {distance})" if distance is not None else ""))
            self.clear_btn.configure(state="normal", bg="#ff5252")
            # start flashing
            self._start_flashing()
            # start buzzer
            self._start_buzzer()
        self.root.after(0, gui_update)

    def _clear_match_ui(self):
        # update UI to no match
        def gui_update():
            self.match_label.configure(text="✅ No Match Detected", fg="#00c853", bg="#111")
            self.person_label.configure(text="")
            self.clear_btn.configure(state="disabled", bg="#444")
            self._stop_flashing()
            self._stop_buzzer()
        self.root.after(0, gui_update)

    def _start_flashing(self):
        # flashing implemented by toggling background colors
        self._flash_on = True
        def flash():
            if getattr(self, "_flash_on", False):
                current = self.match_label.cget("bg")
                # toggle
                new = "#880000" if current != "#880000" else "#440000"
                self.match_label.configure(bg=new)
                self.person_label.configure(bg=new)
                self.right_frame.configure(bg=new)
                self.root.after(500, flash)
        self._flash_on = True
        self.root.after(0, flash)

    def _stop_flashing(self):
        self._flash_on = False
        # restore colors
        self.right_frame.configure(bg="#111")
        self.match_label.configure(bg="#111", fg="#00c853")
        self.person_label.configure(bg="#111")

    def _start_buzzer(self):
        if self.alarm_playing:
            return
        self.alarm_stop_event.clear()
        self.alarm_playing = True

        def buzzer_loop():
            if _audio_player == "simpleaudio" and buzzer_wave is not None:
                # loop by playing and checking stop event
                while not self.alarm_stop_event.is_set():
                    play_obj = buzzer_wave.play()
                    # wait while playing or until stop requested
                    while play_obj.is_playing() and not self.alarm_stop_event.is_set():
                        time.sleep(0.1)
                    # short gap
                    time.sleep(0.2)
            elif _audio_player == "winsound":
                # Win32 PlaySound loop until stopped
                try:
                    # PlaySound with SND_LOOP | SND_ASYNC loops until PlaySound(None) called
                    winsound.PlaySound(BUZZER_WAV, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                    while not self.alarm_stop_event.is_set():
                        time.sleep(0.2)
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception as e:
                    print("[WARN] winsound failed:", e)
            else:
                # fallback visual only (no sound)
                while not self.alarm_stop_event.is_set():
                    time.sleep(0.5)
            self.alarm_playing = False

        threading.Thread(target=buzzer_loop, daemon=True).start()

    def _stop_buzzer(self):
        if not self.alarm_playing:
            return
        self.alarm_stop_event.set()
        # small delay to let thread exit
        time.sleep(0.1)
        self.alarm_playing = False

    def clear_alarm(self):
        # operator clears alarm
        print("[INFO] Alarm cleared by operator.")
        self._clear_match_ui()

    def shutdown(self):
        print("[INFO] Shutting down client app.")
        self.running = False
        self._stop_buzzer()
        time.sleep(0.2)
        try:
            self.vs.release()
        except:
            pass
        self.root.quit()


if __name__ == "__main__":
    root = Tk()
    app = NeuralHashClientApp(root)
    try:
        root.protocol("WM_DELETE_WINDOW", app.shutdown)
        root.mainloop()
    except KeyboardInterrupt:
        app.shutdown()
