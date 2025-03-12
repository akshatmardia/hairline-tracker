import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from HairlineTracker import HairlineTracker

class HairlineTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hairline Tracker")
        self.root.geometry("900x600")
        
        # init tracker
        self.tracker = HairlineTracker()
        
        # create GUI elements
        self.create_widgets()
        
        # update status
        self.update_status()
    
    def create_widgets(self):
        # main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # left panel (controls)
        left_panel = tk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # right panel (display)
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # add image button
        add_btn = tk.Button(left_panel, text="Add New Image", command=self.add_image)
        add_btn.pack(fill=tk.X, pady=5)
        
        # analyze button
        analyze_btn = tk.Button(left_panel, text="Analyze Progress", command=self.show_analysis)
        analyze_btn.pack(fill=tk.X, pady=5)
        
        # compare button
        compare_btn = tk.Button(left_panel, text="Compare Images", command=self.compare_images)
        compare_btn.pack(fill=tk.X, pady=5)
        
        # status frame
        status_frame = tk.LabelFrame(left_panel, text="Status")
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(status_frame, text="No images yet", justify=tk.LEFT, wraplength=280)
        self.status_label.pack(padx=5, pady=5)
        
        # image list frame
        images_frame = tk.LabelFrame(left_panel, text="Tracked Images")
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_listbox = tk.Listbox(images_frame)
        self.image_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # display frame
        display_frame = tk.LabelFrame(right_panel, text="Image View")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # info frame
        info_frame = tk.LabelFrame(right_panel, text="Information")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_label = tk.Label(info_frame, text="Select an image to view details", 
                                  justify=tk.LEFT, wraplength=550)
        self.info_label.pack(padx=5, pady=5, fill=tk.X)
    
    def update_status(self):
        timestamps = self.tracker.get_all_timestamps()
        
        # update status label
        if not timestamps:
            self.status_label.config(text="No images yet. Add your first image to start tracking.")
        else:
            analysis = self.tracker.analyze_progress()
            if "message" in analysis:
                self.status_label.config(text=f"{len(timestamps)} images tracked.\n{analysis['message']}")
            else:
                status_text = f"{len(timestamps)} images tracked.\n"
                status_text += f"Overall: {analysis['overall_direction']}\n"
                status_text += f"Change: {analysis['overall_change_percent']:.2f}%"
                self.status_label.config(text=status_text)
        
        # update image listbox
        self.image_listbox.delete(0, tk.END)
        for ts in timestamps:
            date_part = ts.split("_")[0]
            self.image_listbox.insert(tk.END, date_part)
    
    def add_image(self):
        # add image first choose path
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        # if found path add image to tracker
        if file_path:
            result = self.tracker.add_image(file_path)
            if result:
                self.update_status()
                # select the newly added image
                timestamps = self.tracker.get_all_timestamps()
                if timestamps:
                    self.image_listbox.selection_set(len(timestamps) - 1)
                    self.on_image_select(None)
            else:
                tk.messagebox.showerror("Error", "Failed to process image")
    
    def on_image_select(self, event):
        # image selection from listbox
        selection = self.image_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        timestamps = self.tracker.get_all_timestamps()
        if index < len(timestamps):
            self.display_image(timestamps[index])
    
    def display_image(self, timestamp):
        entry = next((e for e in self.tracker.tracking_data["entries"] if e["timestamp"] == timestamp), None)
        if not entry:
            return
        
        # display image
        img = cv2.imread(entry["processed_path"])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # resize
            display_width = 600
            ratio = display_width / pil_img.width
            display_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((display_width, display_height))
            
            # convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # keep reference image
            
            # image info
            date_part = timestamp.split("_")[0]
            time_part = timestamp.split("_")[1].replace("-", ":")
            avg_distance = np.mean(entry["hairline_distances"]) if entry["hairline_distances"] else 0
            
            info_text = f"Date: {date_part} {time_part}\n"
            info_text += f"Average hairline distance: {avg_distance:.2f} pixels\n"
            
            # add change information if a previous image exists
            timestamps = self.tracker.get_all_timestamps()
            idx = timestamps.index(timestamp)
            if idx > 0:
                prev_ts = timestamps[idx - 1]
                prev_entry = next((e for e in self.tracker.tracking_data["entries"] if e["timestamp"] == prev_ts), None)
                if prev_entry and prev_entry["hairline_distances"]:
                    prev_avg = np.mean(prev_entry["hairline_distances"])
                    change = avg_distance - prev_avg
                    change_percent = (change / prev_avg) * 100
                    direction = "receding" if change < 0 else "advancing" if change > 0 else "stable"
                    
                    info_text += f"Since previous image: {direction}\n"
                    info_text += f"Change: {change:.2f} pixels ({change_percent:.2f}%)"
            
            self.info_label.config(text=info_text)
    
    def show_analysis(self):
        # analysis results
        chart_path = self.tracker.generate_progress_chart()
        if not chart_path:
            tk.messagebox.showinfo("Analysis", "Need at least 2 images to analyze progress.")
            return
        
        # analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Hairline Progress Analysis")
        analysis_window.geometry("800x600")
        
        # display chart
        img = Image.open(chart_path)
        photo = ImageTk.PhotoImage(img)
        
        chart_label = tk.Label(analysis_window, image=photo)
        chart_label.image = photo  # keep reference image
        chart_label.pack(padx=10, pady=10)
        
        # analysis text
        analysis = self.tracker.analyze_progress()
        
        text_frame = tk.Frame(analysis_window)
        text_frame.pack(fill=tk.X, padx=10, pady=10)
        
        analysis_text = f"Analysis period: {analysis['period_start']} to {analysis['period_end']}\n\n"
        analysis_text += f"Overall trend: {analysis['overall_direction']}\n"
        analysis_text += f"Total change: {analysis['overall_change_pixels']:.2f} pixels "
        analysis_text += f"({analysis['overall_change_percent']:.2f}%)\n\n"
        
        analysis_text += "Change by measurement:\n"
        for change in analysis["changes"]:
            analysis_text += f"- {change['from']} to {change['to']}: "
            analysis_text += f"{change['direction']} by {change['change_pixels']:.2f} pixels "
            analysis_text += f"({change['change_percent']:.2f}%)\n"
        
        analysis_label = tk.Label(text_frame, text=analysis_text, justify=tk.LEFT)
        analysis_label.pack(anchor=tk.W)
    
    def compare_images(self):
        # compare two images
        timestamps = self.tracker.get_all_timestamps()
        if len(timestamps) < 2:
            tk.messagebox.showinfo("Compare", "Need at least 2 images to compare.")
            return
        
        # window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Compare Images")
        compare_window.geometry("350x200")
        
        # select first image
        tk.Label(compare_window, text="First image:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        first_var = tk.StringVar()
        first_combo = tk.OptionMenu(compare_window, first_var, *[t.split("_")[0] for t in timestamps])
        first_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # select second image
        tk.Label(compare_window, text="Second image:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        second_var = tk.StringVar()
        second_combo = tk.OptionMenu(compare_window, second_var, *[t.split("_")[0] for t in timestamps])
        second_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        # set default time
        first_var.set(timestamps[0].split("_")[0])
        second_var.set(timestamps[-1].split("_")[0])
        
        # compare button
        def do_compare():
            # get timestamp
            first_date = first_var.get()
            second_date = second_var.get()
            
            first_ts = next((t for t in timestamps if t.startswith(first_date)), None)
            second_ts = next((t for t in timestamps if t.startswith(second_date)), None)
            
            if not first_ts or not second_ts or first_ts == second_ts:
                tk.messagebox.showwarning("Compare", "Please select two different images.")
                return
            
            comparison_path = self.tracker.create_comparison_image(first_ts, second_ts)
            if not comparison_path:
                tk.messagebox.showerror("Error", "Failed to create comparison.")
                return
            
            # display comparison
            result_window = tk.Toplevel(compare_window)
            result_window.title(f"Comparison: {first_date} vs {second_date}")
            
            # display image
            img = cv2.imread(comparison_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # resize
            display_width = min(1200, pil_img.width)
            ratio = display_width / pil_img.width
            display_height = int(pil_img.height * ratio)
            pil_img = pil_img.resize((display_width, display_height))
            
            # convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # display in window
            label = tk.Label(result_window, image=photo)
            label.image = photo  # keep reference image
            label.pack(padx=10, pady=10)
            
            # compare and analyze
            first_entry = next((e for e in self.tracker.tracking_data["entries"] if e["timestamp"] == first_ts), None)
            second_entry = next((e for e in self.tracker.tracking_data["entries"] if e["timestamp"] == second_ts), None)
            
            if first_entry and second_entry:
                first_avg = np.mean(first_entry["hairline_distances"]) if first_entry["hairline_distances"] else 0
                second_avg = np.mean(second_entry["hairline_distances"]) if second_entry["hairline_distances"] else 0
                
                # difference
                change = second_avg - first_avg
                change_percent = (change / first_avg) * 100 if first_avg else 0
                direction = "Receding" if change < 0 else "Advancing" if change > 0 else "Stable"
                
                analysis_text = f"Analysis: {direction} hairline\n"
                analysis_text += f"Change: {abs(change):.2f} pixels ({abs(change_percent):.2f}%)"
                
                analysis_label = tk.Label(result_window, text=analysis_text, font=("Arial", 12, "bold"))
                analysis_label.pack(pady=10)
        
        compare_btn = tk.Button(compare_window, text="Compare", command=do_compare)
        compare_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=15)
