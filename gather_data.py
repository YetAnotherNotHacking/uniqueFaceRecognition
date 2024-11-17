import os
import cv2
import tkinter as tk
from tkinter import messagebox

class ImageCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Capture GUI")

        self.label = tk.Label(root, text="Enter Username:")
        self.label.pack(pady=10)

        self.entry = tk.Entry(root)
        self.entry.pack(pady=10)

        self.capture_btn = tk.Button(root, text="Capture Images", command=self.start_capture)
        self.capture_btn.pack(pady=10)

        self.cap = None
        self.username = ""

    def start_capture(self):
        self.username = self.entry.get().strip()

        if not self.username:
            messagebox.showerror("Error", "Please enter a valid username.")
            return

        save_path = os.path.join("dataset", self.username)
        os.makedirs(save_path, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        count = 1

        while count <= 255:
            ret, frame = self.cap.read()

            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                break

            frame = cv2.flip(frame, 1)
            cv2.imshow("Capture Images - Press 'q' to Quit", frame)

            img_name = os.path.join(save_path, f"{str(count).zfill(2)}.png")
            cv2.imwrite(img_name, frame)

            count += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        if count > 255:
            messagebox.showinfo("Info", f"~255 images saved to /dataset/{self.username}")
        else:
            messagebox.showinfo("Info", f"Captured {count - 1} images.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptureApp(root)
    root.mainloop()
