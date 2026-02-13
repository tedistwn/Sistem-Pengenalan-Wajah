import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Pengenalan Wajah")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        self.users = self.load_users()
        self.is_running = False
        self.video_label = None
        self.ui_elements = [] 
        
        self.setup_ui()
        
    def load_users(self):
        users = {}
        if os.path.exists('users.txt'):
            try:
                with open('users.txt', 'r') as f:
                    for line in f:
                        id_str, name = line.strip().split(',', 1)
                        users[int(id_str)] = name
            except:
                pass
        return users
    
    def save_users(self):
        with open('users.txt', 'w') as f:
            for uid, name in self.users.items():
                f.write(f"{uid},{name}\n")
    
    def setup_ui(self):
        tk.Label(
            self.root, text="🎭 SISTEM PENGENALAN WAJAH",
            font=("Times New Roman", 22, "bold"), bg='#34495e', fg='white', pady=12
        ).pack(fill='x')
        
        main = tk.Frame(self.root, bg='#2c3e50')
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        left = tk.Frame(main, bg='#34495e', width=280)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        
        self.info_label = tk.Label(
            left, text=f"📊 Pengguna: {len(self.users)}",
            font=("Times New Roman", 11), bg='#34495e', fg='white', pady=8
        )
        self.info_label.pack(pady=10)
        
        btn_config = {
            'font': ("Times New Roman", 10, "bold"),
            'fg': 'white',
            'bd': 0,
            'pady': 10,
            'cursor': 'hand2'
        }

        buttons = [
            ("➕ Tambah Pengguna", self.add_user, '#3498db'),
            ("🎓 Training Model", self.train_model, '#3498db'),
            ("🔍 Pengenalan Wajah", self.start_recognition, '#3498db'),
            ("👥 Kelola Pengguna", self.manage_users, '#95a5a6'),
            ("❌ Keluar", self.quit_app, '#e74c3c'),
        ]
        
        for text, cmd, color in buttons[:-1]:
            tk.Button(left, text=text, command=cmd, bg=color, 
                     activebackground=color, **btn_config).pack(fill='x', padx=12, pady=4)
        
        tk.Button(left, text=buttons[-1][0], command=buttons[-1][1], 
                 bg=buttons[-1][2], activebackground='#c0392b', 
                 **btn_config).pack(fill='x', padx=12, pady=4, side='bottom')
        
        right = tk.Frame(main, bg='#34495e')
        right.pack(side='right', fill='both', expand=True)
        
        self.display_frame = tk.Frame(right, bg='black')
        self.display_frame.pack(fill='both', expand=True, padx=8, pady=8)
        
        self.welcome_label = tk.Label(
            self.display_frame,
            text="Selamat Datang!\n\nPilih menu di sebelah kiri",
            font=("Times New Roman", 15), bg='black', fg='white'
        )
        self.welcome_label.place(relx=0.5, rely=0.5, anchor='center')
        
        self.status_bar = tk.Label(
            self.root, text="Status: Siap",
            font=("Times New Roman", 9), bg='#34495e', fg='white',
            anchor='w', padx=8, pady=4
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def clear_display(self):
        if self.welcome_label:
            self.welcome_label.destroy()
            self.welcome_label = None
        
        if self.video_label:
            self.video_label.config(image='')
            self.video_label.image = None
        
        for elem in self.ui_elements:
            try:
                elem.destroy()
            except:
                pass
        self.ui_elements.clear()
    
    def update_status(self, msg):
        self.status_bar.config(text=f"Status: {msg}")
        self.root.update()
    
    def add_user(self):
        d = tk.Toplevel(self.root)
        d.title("Tambah Pengguna")
        d.geometry("380x300")
        d.configure(bg='#34495e')
        d.transient(self.root)
        d.grab_set()
        d.resizable(False, False)
        
        d.update_idletasks()
        x = (d.winfo_screenwidth() - 380) // 2
        y = (d.winfo_screenheight() - 300) // 2
        d.geometry(f"+{x}+{y}")
        
        header = tk.Frame(d, bg='#2c3e50', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="📸 Tambah Pengguna Baru", font=("Times New Roman", 14, "bold"),
                bg='#2c3e50', fg='white').pack(expand=True)
        
        frame = tk.Frame(d, bg='#34495e')
        frame.pack(pady=20, padx=35, fill='both', expand=True)

        tk.Label(frame, text="Nama:", bg='#34495e', fg='white', 
                font=("Times New Roman", 10, "bold")).pack(pady=(0,5))
        name_entry = tk.Entry(frame, font=("Times New Roman", 11), width=26, justify='center')
        name_entry.pack(ipady=5, pady=(0,15))
        name_entry.focus()

        tk.Label(frame, text="Jumlah Foto (20-1000):", bg='#34495e', fg='white',
                font=("Times New Roman", 10, "bold")).pack(pady=(0,5))

        photo_count = tk.Spinbox(frame, from_=20, to=1000, font=("Times New Roman", 11), 
                                width=10, justify='center')
        photo_count.delete(0, tk.END)
        photo_count.insert(0, "80")
        photo_count.pack(ipady=4, pady=(0,5))

        tk.Label(frame, text="Rekomendasi: 80-150", bg='#34495e', 
                fg='#bdc3c7', font=("Times New Roman", 9, "italic")).pack()
        
        def start():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Peringatan", "Nama tidak boleh kosong!")
                return
            
            try:
                total = int(photo_count.get())
                if not (20 <= total <= 1000):
                    raise ValueError()
            except:
                messagebox.showwarning("Peringatan", "Jumlah foto: 20-1000!")
                return
            
            new_id = max(self.users.keys()) + 1 if self.users else 1
            d.destroy()
            self.collect_dataset(name, new_id, total)
        
        name_entry.bind('<Return>', lambda e: start())
        photo_count.bind('<Return>', lambda e: start())

        btn_frame = tk.Frame(d, bg='#34495e')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="✓ Mulai", command=start,
                 font=("Times New Roman", 10, "bold"), bg='#27ae60', fg='white',
                 width=10, pady=8, cursor='hand2', bd=0).pack(side='left', padx=5)
        tk.Button(btn_frame, text="✗ Batal", command=d.destroy,
                 font=("Times New Roman", 10, "bold"), bg='#e74c3c', fg='white',
                 width=10, pady=8, cursor='hand2', bd=0).pack(side='left', padx=5)
    
    def collect_dataset(self, name, user_id, total_photos):
        if self.is_running:
            messagebox.showwarning("Peringatan", "Proses sedang berjalan!")
            return
        
        self.is_running = True
        self.clear_display()
        self.update_status(f"Mengambil {total_photos} foto: {name}")
        
        if not self.video_label:
            self.video_label = tk.Label(self.display_frame, bg='black')
            self.video_label.pack(fill='both', expand=True)
        
        info = tk.Frame(self.display_frame, bg='#2c3e50', height=70)
        info.place(x=0, y=0, relwidth=1)
        self.ui_elements.append(info)
        
        tk.Label(info, text=f"👤 {name}", font=("Times New Roman", 12, "bold"),
                bg='#2c3e50', fg='white').pack(pady=(8,0))
        
        progress = tk.Label(info, text=f"📸 0 / {total_photos}", 
                           font=("Times New Roman", 14, "bold"), bg='#2c3e50', fg='#00ff00')
        progress.pack(pady=(2,0))
        
        def worker():
            try:
                os.makedirs('dataset', exist_ok=True)
                
                cam = cv2.VideoCapture(0)
                cam.set(3, 640)
                cam.set(4, 480)
                
                if not cam.isOpened():
                    raise Exception("Gagal membuka kamera!")
                
                detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                count = 0
                skip = 0
                
                while self.is_running and count < total_photos:
                    ret, img = cam.read()
                    if not ret:
                        break
                    
                    skip += 1
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    if skip % 2 == 0:
                        faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(50,50))
                        
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            if skip % 4 == 0 and count < total_photos:
                                count += 1
                                face_crop = gray[y:y+h, x:x+w]
                                face_crop = cv2.resize(face_crop, (400, 400), interpolation=cv2.INTER_CUBIC)
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                face_crop = clahe.apply(face_crop)
                                face_crop = cv2.GaussianBlur(face_crop, (3, 3), 0)

                                filename = f"dataset/{name}.{user_id}.{count}.jpg"
                                cv2.imwrite(filename, face_crop)
                                
                                pct = int((count / total_photos) * 100)
                                self.root.after(0, lambda c=count, p=pct: 
                                    progress.config(text=f"📸 {c}/{total_photos} ({p}%)"))
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    self.root.after(0, lambda i=img_tk: self._update_video(i))
                
                cam.release()
                
                if count >= total_photos:
                    self.users[user_id] = name
                    self.save_users()
                    self.root.after(0, lambda: self.info_label.config(
                        text=f"📊 Pengguna: {len(self.users)}"))
                    self.root.after(0, lambda: messagebox.showinfo("Sukses",
                        f"✓ Berhasil: {count} foto untuk {name}!\n\n"
                        f"Format file: {name}.{user_id}.*.jpg\n\n"
                        f"Lanjutkan dengan TRAINING MODEL."))
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            finally:
                self.is_running = False
                self.root.after(0, self.clear_display)
                self.root.after(0, lambda: self.update_status("Siap"))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _update_video(self, img_tk):
        if self.video_label:
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
    
    def train_model(self):
        if self.is_running:
            messagebox.showwarning("Peringatan", "Proses sedang berjalan!")
            return
        
        if not os.path.exists('dataset') or not os.listdir('dataset'):
            messagebox.showwarning("Peringatan", "Dataset kosong!\nTambahkan pengguna dulu.")
            return
        
        if not messagebox.askyesno("Konfirmasi", "Mulai training model?"):
            return
        
        self.is_running = True
        self.update_status("Training model...")
        
        def worker():
            try:
                os.makedirs('trainer', exist_ok=True)
                
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                
                paths = [os.path.join('dataset', f) for f in os.listdir('dataset') 
                        if f.endswith('.jpg')]
                
                faces, ids = [], []
                for path in paths:
                    try:
                        img = Image.open(path).convert('L')
                        img_np = np.array(img, 'uint8')
                        
                        filename = os.path.basename(path)
                        parts = filename.split(".")
                        id = int(parts[1]) 
                        
                        faces.append(img_np)
                        ids.append(id)
                    except Exception as e:
                        print(f"Skip {path}: {e}")
                        continue
                
                if not faces:
                    raise Exception("Tidak ada wajah terdeteksi!")
                
                recognizer.train(faces, np.array(ids))
                recognizer.write('trainer/trainer.yml')
                
                with open('trainer/metadata.txt', 'w') as f:
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Samples: {len(faces)}\n")
                    f.write(f"Users: {len(np.unique(ids))}\n")
                
                self.root.after(0, lambda: messagebox.showinfo("Sukses",
                    f"Training selesai!\n\n{len(np.unique(ids))} pengguna\n"
                    f"{len(faces)} sampel"))
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            finally:
                self.is_running = False
                self.root.after(0, lambda: self.update_status("Siap"))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def start_recognition(self):
        if self.is_running:
            messagebox.showwarning("Peringatan", "Proses sedang berjalan!")
            return
        
        if not os.path.exists('trainer/trainer.yml'):
            messagebox.showwarning("Peringatan", "Model belum dilatih!")
            return
        
        self.is_running = True
        self.clear_display()
        self.update_status("Pengenalan wajah aktif")
        
        if not self.video_label:
            self.video_label = tk.Label(self.display_frame, bg='black')
            self.video_label.pack(fill='both', expand=True)
        
        stop_btn = tk.Button(
            self.display_frame, text="⏹ Stop",
            command=self.stop_recognition,
            font=("Times New Roman", 11, "bold"), bg='#e74c3c', fg='white',
            padx=18, pady=8
        )
        stop_btn.place(relx=0.5, rely=0.93, anchor='center')
        self.ui_elements.append(stop_btn)
        
        def worker():
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8
                )
                recognizer.read('trainer/trainer.yml')
                
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                
                names = {0: 'Unknown'}
                names.update(self.users)
                
                cam = cv2.VideoCapture(0)
                cam.set(3, 640)
                cam.set(4, 480)
                
                if not cam.isOpened():
                    raise Exception("Gagal membuka kamera!")
                
                from collections import Counter
                recent_predictions = []
                
                while self.is_running:
                    ret, img = cam.read()
                    if not ret:
                        break
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60,60))
                    
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (400, 400), interpolation=cv2.INTER_CUBIC)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        face_roi = clahe.apply(face_roi)
                        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
                        id, conf = recognizer.predict(face_roi)

                        recent_predictions.append((id, conf))
                        if len(recent_predictions) > 5:
                            recent_predictions.pop(0)

                        valid = [(i, c) for i, c in recent_predictions if c < 40]
                        if valid:
                            most_common_id = Counter(i for i, c in valid).most_common(1)[0][0]
                            avg_conf = sum(c for i, c in valid if i == most_common_id) / len(valid)
                            id = most_common_id
                            conf = avg_conf

                        if conf < 40:
                            name = names.get(id, "Unknown")
                            conf_score = round(100 - conf)
                            conf_text = f"{conf_score}%"
                            color = (0,255,0) if conf < 20 else (0,255,255) if conf < 40 else (0,165,255)
                        else:
                            name, conf_text, color = "Unknown", "0%", (0,0,255)
                        
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.rectangle(img, (x, y-30), (x+w, y), color, -1)
                        cv2.putText(img, name, (x+5, y-8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        cv2.putText(img, conf_text, (x+5, y+h-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    
                    self.root.after(0, lambda i=img_tk: self._update_video(i))
                
                cam.release()
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            finally:
                self.is_running = False
                self.root.after(0, self.clear_display)
                self.root.after(0, lambda: self.update_status("Siap"))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def stop_recognition(self):
        self.is_running = False
        self.update_status("Menghentikan...")
    
    def manage_users(self):
        d = tk.Toplevel(self.root)
        d.title("Kelola Pengguna")
        d.geometry("480x380")
        d.configure(bg='#34495e')
        
        tk.Label(d, text="Daftar Pengguna", font=("Times New Roman", 13, "bold"),
                bg='#34495e', fg='white', pady=10).pack()
        
        frame = tk.Frame(d, bg='#34495e')
        frame.pack(fill='both', expand=True, padx=15, pady=8)
        
        scroll = tk.Scrollbar(frame)
        scroll.pack(side='right', fill='y')
        
        listbox = tk.Listbox(frame, font=("Times New Roman", 10),
                            yscrollcommand=scroll.set, bg='#ecf0f1')
        listbox.pack(fill='both', expand=True)
        scroll.config(command=listbox.yview)
        
        for uid, name in sorted(self.users.items()):
            listbox.insert(tk.END, f"ID {uid}: {name}")
        
        def delete():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Peringatan", "Pilih pengguna!")
                return
            
            item = listbox.get(sel[0])
            uid = int(item.split(":")[0].replace("ID ", ""))
            name = self.users[uid]
            
            if messagebox.askyesno("Konfirmasi", 
                f"Hapus '{name}'?\n\nSemua foto akan dihapus."):
                
                del self.users[uid]
                self.save_users()

                if os.path.exists('dataset'):
                    for f in os.listdir('dataset'):
                        try:
                            parts = f.split(".")
                            if len(parts) >= 3 and parts[1] == str(uid):
                                os.remove(os.path.join('dataset', f))
                                print(f"Deleted: {f}")
                        except Exception as e:
                            print(f"Error deleting {f}: {e}")
                            pass
                
                listbox.delete(sel[0])
                self.info_label.config(text=f"📊 Pengguna: {len(self.users)}")
                messagebox.showinfo("Sukses", f"'{name}' dihapus!")
        
        def delete_trainer():
            trainer_path = 'trainer/trainer.yml'
            if not os.path.exists(trainer_path):
                messagebox.showinfo("Info", "File trainer.yml tidak ditemukan!")
                return
            
            if messagebox.askyesno("Konfirmasi", 
                "Hapus file trainer.yml?\n\nModel akan dihapus dan perlu dilatih ulang."):
                try:
                    os.remove(trainer_path)
                    messagebox.showinfo("Sukses", "File trainer.yml berhasil dihapus!")
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menghapus: {str(e)}")
        
        btn_frame = tk.Frame(d, bg='#34495e')
        btn_frame.pack(pady=8)
        
        tk.Button(btn_frame, text="Hapus", command=delete,
                 font=("Times New Roman", 10), bg='#e74c3c', fg='white',
                 padx=15, pady=6).pack(side='left', padx=4)
        tk.Button(btn_frame, text="Hapus trainer.yml", command=delete_trainer,
                 font=("Times New Roman", 10), bg='#e67e22', fg='white',
                 padx=15, pady=6).pack(side='left', padx=4)
        tk.Button(btn_frame, text="Tutup", command=d.destroy,
                 font=("Times New Roman", 10), bg='#95a5a6', fg='white',
                 padx=15, pady=6).pack(side='left', padx=4)
    
    def quit_app(self):
        if self.is_running:
            if not messagebox.askyesno("Konfirmasi", 
                "Proses sedang berjalan!\n\nYakin keluar?"):
                return
            self.is_running = False
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_app)
    root.mainloop()

if __name__ == '__main__':
    main()