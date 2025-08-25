# File: 4_interactive_review_tool.py
# Purpose: Provides a GUI to review suspicious images against a reference image,
#          allowing the user to delete the suspicious one if it's a mismatch.

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import json
import os

# --- Configuration ---
CSV_FILE = "review_list2.csv"
JSON_FILE = "reduced_dataset2.json"
# --- This should be the same path you used in your image generation script ---
CROPPED_FACES_PATH = "C:\\Users\\ASUS\\Desktop\\nFilterd_cropped_faces"

class ImageReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Review Tool")

        try:
            self.review_data = pd.read_csv(CSV_FILE).to_dict('records')
            with open(JSON_FILE, 'r') as f:
                self.dataset = json.load(f)
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not find a necessary file: {e.filename}")
            self.root.destroy()
            return

        if not self.review_data:
            messagebox.showinfo("Complete", "No images to review in the CSV file.")
            self.root.destroy()
            return

        self.current_index = 0

        # --- GUI Elements ---
        self.info_label = tk.Label(root, text="", font=("Helvetica", 12), justify=tk.LEFT)
        self.info_label.pack(pady=10)

        # Main frame to hold both images side-by-side
        images_frame = tk.Frame(root)
        images_frame.pack(pady=10, padx=20, fill="x", expand=True)

        # Left side: Reference Image
        ref_frame = tk.Frame(images_frame)
        ref_frame.pack(side=tk.LEFT, padx=10, expand=True)
        tk.Label(ref_frame, text="Reference Image", font=("Helvetica", 10, "bold")).pack()
        self.ref_image_label = tk.Label(ref_frame)
        self.ref_image_label.pack()

        # Right side: Suspicious Image
        culprit_frame = tk.Frame(images_frame)
        culprit_frame.pack(side=tk.RIGHT, padx=10, expand=True)
        tk.Label(culprit_frame, text="Suspicious Image", font=("Helvetica", 10, "bold")).pack()
        self.culprit_image_label = tk.Label(culprit_frame)
        self.culprit_image_label.pack()

        self.progress_label = tk.Label(root, text="", font=("Helvetica", 10))
        self.progress_label.pack(pady=5)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)

        self.keep_button = tk.Button(button_frame, text="Keep", width=15, command=self.keep_image, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        self.keep_button.pack(side=tk.LEFT, padx=10)

        self.delete_button = tk.Button(button_frame, text="Delete Suspicious Image", width=20, command=self.delete_image, bg="#F44336", fg="white", font=("Helvetica", 10, "bold"))
        self.delete_button.pack(side=tk.LEFT, padx=10)

        self.load_image()

    def _load_and_display_image(self, label_widget, image_path, size=(300, 300)):
        """Helper function to load an image and place it in a Tkinter label."""
        try:
            img = Image.open(image_path)
            img.thumbnail(size)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo # Keep a reference to avoid garbage collection
        except FileNotFoundError:
            label_widget.config(text=f"Image not found:\n{os.path.basename(image_path)}", image='')
            label_widget.image = None

    def load_image(self):
        """Loads and displays the current reference and suspicious images."""
        if self.current_index >= len(self.review_data):
            self.finish_review()
            return

        self.current_item = self.review_data[self.current_index]
        folder = self.current_item['FolderName']
        culprit_filename = self.current_item['FileName']
        dissimilarity = self.current_item['AvgHammingDissimilarity']

        # Update text info
        info_text = f"Person Folder: {folder}\nSuspicious File: {culprit_filename}\nDissimilarity Score: {dissimilarity:.2f}"
        self.info_label.config(text=info_text)
        progress_text = f"Reviewing Image {self.current_index + 1} of {len(self.review_data)}"
        self.progress_label.config(text=progress_text)

        # --- Find and load REFERENCE image ---
        ref_filename = None
        person_images = self.dataset.get(folder, [])
        if person_images:
            # Use the first image in the folder as reference
            ref_filename = person_images[0]['filename']
            # If the first image IS the suspicious one, try to use the second image instead
            if ref_filename == culprit_filename and len(person_images) > 1:
                ref_filename = person_images[1]['filename']

        if ref_filename:
            ref_image_path = os.path.join(CROPPED_FACES_PATH, folder, ref_filename)
            self._load_and_display_image(self.ref_image_label, ref_image_path)
        else:
            self.ref_image_label.config(text="No reference found", image='')

        # --- Find and load CULPRIT image ---
        culprit_image_path = os.path.join(CROPPED_FACES_PATH, folder, culprit_filename)
        self._load_and_display_image(self.culprit_image_label, culprit_image_path)

    def keep_image(self):
        """Moves to the next image without making changes."""
        self.current_index += 1
        self.load_image()

    def delete_image(self):
        """Deletes ONLY the suspicious image file and its corresponding JSON entry."""
        folder = self.current_item['FolderName']
        filename = self.current_item['FileName']

        # 1. Delete the suspicious image file
        image_path = os.path.join(CROPPED_FACES_PATH, folder, filename)
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"🗑️ Deleted image file: {image_path}")
        except OSError as e:
            messagebox.showerror("Error", f"Error deleting image file: {e}")
            return

        # 2. Remove the hash entity from the JSON data
        if folder in self.dataset:
            original_length = len(self.dataset[folder])
            self.dataset[folder] = [img for img in self.dataset[folder] if img['filename'] != filename]
            if len(self.dataset[folder]) < original_length:
                 print(f"🗑️ Removed JSON entry for: {filename} in folder {folder}")
            else:
                 print(f"⚠️ Could not find JSON entry for: {filename} in folder {folder}")

        self.current_index += 1
        self.load_image()

    def save_json_changes(self):
        """Saves the modified dataset back to the JSON file."""
        try:
            with open(JSON_FILE, 'w') as f:
                json.dump(self.dataset, f, indent=2)
            print(f"✅ Successfully saved updated data to '{JSON_FILE}'.")
        except IOError as e:
            messagebox.showerror("Save Error", f"Could not save changes to JSON file: {e}")

    def finish_review(self):
        """Called when all images have been reviewed."""
        self.save_json_changes()
        messagebox.showinfo("Complete", "Review finished. All changes have been saved.")
        self.root.destroy()

# --- Main Application Start ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewer(root)
    root.mainloop()