import os
from datetime import datetime
import pickle
import json
import cv2
import numpy as np
import dlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj) # NumpyEncoder, self

class HairlineTracker:
    def __init__(self, storage_directory='hairline_data'):
        # init a storage directory
        self.storage_dir = storage_directory
        self.images_dir = os.path.join(storage_directory, 'images')
        self.data_path = os.path.join(storage_directory, 'tracking_data.json')
        self.landmark_data_path = os.path.join(storage_directory, 'landmark_data.pkl')

        # create dirs if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)

        # load existing data if available
        self.tracking_data = self.load_tracking_data()
        self.landmark_data = self.load_landmark_data()

        # init facial detection tools
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor_path = os.path.join(storage_directory,
                                           'shape_predictor_68_face_landmarks.dat')

        # prompt to download facial landmark predictor if not present
        if not os.path.exists(self.predictor_path):
            print("Facial landmark predictor not found. Please download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"And extract it to: {self.predictor_path}")
            print("After downloading, restart the application.")
            return

        self.predictor = dlib.shape_predictor(self.predictor_path)

    def load_tracking_data(self):
        # load old tracking data
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                return json.load(f)
        return {"entries": []}

    def load_landmark_data(self):
        # load landmark data
        if os.path.exists(self.landmark_data_path):
            with open(self.landmark_data_path, 'rb') as f:
                return pickle.load(f) # using pickle file to serialize
        return {}

    def save_tracking_data(self):
        # open json file and save data there
        with open(self.data_path, 'w') as f:
            json.dump(self.tracking_data, f, cls=NumpyEncoder, indent=4)

    def save_landmark_data(self):
        # open pickle file and save data there
        with open(self.landmark_data_path, 'wb') as f:
            pickle.dump(self.landmark_data, f)

    def add_image(self, image_path):
        # timestamp for image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        # process image
        processed_data = self.process_image(img, timestamp)
        if processed_data is None:
            print("Error: Could not process image")
            return None

        # save processed image
        saved_path = os.path.join(self.images_dir, f"hairline_{timestamp}.jpg")
        cv2.imwrite(saved_path, processed_data["aligned_image"])

        # create data entry
        entry = {
            "timestamp": timestamp,
            "original_path": image_path,
            "processed_path": saved_path,
            "hairline_distances": processed_data["hairline_distances"],
            "hairline_points": processed_data["hairline_points"],
        }

        # store landmark data separately since it is in a pickle file
        self.landmark_data[timestamp] = {
            "landmarks": processed_data["landmarks"],
            "face_rect": processed_data["face_rect"]
        }

        # add to tracking data
        self.tracking_data["entries"].append(entry)

        # save all data
        self.save_tracking_data()
        self.save_landmark_data()

        return entry

    def process_image(self, img, timestamp):
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # eye detection
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        # if both eyes not found, throw error
        if len(eyes) < 2:
            print("Couldn't detect both eyes. Please ensure eyes are clearly visible.")
            return None

        # sort eyes by x-coord to determine left and right
        eyes = sorted(eyes, key=lambda x: x[0])

        # align image using eyes
        left_eye = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
        right_eye = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)

        # estimate forehead region based on eye positions
        eye_midpoint = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

        # estimate forehead region
        forehead_height = max(int(eye_midpoint[1] - eye_distance * 1.2), 0)  # start 120% of eye_distance above eyes
        forehead_top = max(int(forehead_height - eye_distance * 0.4), 0)  # go up by 40% of eye_distance
        forehead_left = max(int(left_eye[0] - eye_distance * 0.5), 0) # 0.5
        forehead_right = min(int(right_eye[0] + eye_distance * 0.8), img.shape[1]) # 0.9

        # detect hairline
        hairline_points, hairline_img = self.detect_hairline(
            img,
            forehead_top,
            forehead_height,
            forehead_left,
            forehead_right,
            eye_midpoint
        )

        # calculate distances from eye midpoint to each hairline point
        hairline_distances = []
        for point in hairline_points:
            distance = np.sqrt((point[0] - eye_midpoint[0])**2 + (point[1] - eye_midpoint[1])**2)
            hairline_distances.append(float(distance))

        # empty landmarks and face_rect for consistency
        aligned_points = [(0, 0)] * 68  # empty landmarks
        aligned_points[36:42] = [(left_eye[0], left_eye[1])] * 6  # right eye points
        aligned_points[42:48] = [(right_eye[0], right_eye[1])] * 6  # left eye points

        # create a face rectangle based on eye positions
        face_width = int(eye_distance * 2.5)
        face_height = int(eye_distance * 3)
        face_left = max(eye_midpoint[0] - face_width // 2, 0)
        face_top = max(eye_midpoint[1] - face_height // 3, 0)  # position face so eyes are in upper third
        face_rect = (face_left, face_top, face_left + face_width, face_top + face_height)

        # return results
        return {
            "aligned_image": hairline_img,
            "landmarks": aligned_points,
            "face_rect": face_rect,
            "hairline_points": hairline_points,
            "hairline_distances": hairline_distances
        }

    def remove_outliers(self, hairline_points, threshold=1.5):
        # extract y values
        y_values = np.array([point[1] for point in hairline_points])

        # IQR method
        # q1 = np.percentile(y_values, 25)
        # q3 = np.percentile(y_values, 75)
        # iqr = q3 - q1
        # lower_bound = q1 - (threshold * iqr)
        # upper_bound = q3 + (threshold * iqr)

        # STD method
        y_mean = np.mean(y_values)
        y_std = np.std(y_values)
        lower_bound = y_mean - threshold * y_std
        # upper_bound = y_mean + threshold * y_std

        # filter points within bounds
        filtered_points = [point for point in hairline_points if lower_bound <= point[1]]
        return filtered_points

    def detect_hairline(self, img, forehead_top, forehead_bottom, forehead_left, forehead_right, eye_midpoint):
        """Detect the hairline using just eye positions as reference."""
        # image copy
        result_img = img.copy()
        ### TESTING
        # r_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        # r_blur = cv2.GaussianBlur(r_gray, (5, 5), 0)
        # result_img = cv2.Canny(r_blur, 50, 150)

        # draw region of interest for debugging
        cv2.rectangle(result_img, (forehead_left, forehead_top), (forehead_right, forehead_bottom), (0, 255, 255), 2)

        # define forehead ROI
        forehead_roi = img[forehead_top:forehead_bottom, forehead_left:forehead_right]

        # convert to grayscale, blur and use canny edge detection
        forehead_gray = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2GRAY)
        forehead_blurred = cv2.GaussianBlur(forehead_gray, (5, 5), 0) # (5, 5) kernel size works best
        forehead_edges = cv2.Canny(forehead_blurred, 50, 150)

        # find hairline points
        hairline_points = []
        num_columns = 30  # number of sample points (good: 30, 50)
        column_width = forehead_roi.shape[1] // num_columns

        # ignore the first few columns
        left_margin = 4
        right_margin = 4
        threshold = 0

        for i in range(num_columns):
        # for i in range(left_margin, num_columns - right_margin):
            col_center = i * column_width + column_width // 2
            if col_center >= forehead_edges.shape[1]:
                continue

            # debugging column lines
            # cv2.line(result_img, (col_center + forehead_left, forehead_top), (col_center + forehead_left, forehead_bottom), (0, 0, 255), 1)

            column = forehead_edges[:, col_center]

            # dynamically change threshold based on position
            if i < left_margin or i >= num_columns - right_margin:
                threshold = 320  # higher threshold at the ends
            else:
                threshold = 0

            edge_indices = np.where(column > threshold)[0]
            # edge_indices = np.where(column > 0)[0]

            if len(edge_indices) > 0:
                # lowest edge point in this column as hairline point
                hairline_y = edge_indices[-1] + forehead_top
                hairline_x = col_center + forehead_left
                hairline_points.append((hairline_x, hairline_y))

                # draw point
                cv2.circle(result_img, (hairline_x, hairline_y), 3, (0, 255, 0), -1)

        # draw connecting line for detected hairline
        # hairline_points = self.remove_outliers(hairline_points)
        if len(hairline_points) > 1:
            hairline_points.sort(key=lambda p: p[0])  # sort by x-coordinate
            for i in range(len(hairline_points) - 1):
                cv2.line(result_img, hairline_points[i], hairline_points[i + 1], (255, 0, 0), 2)

        return hairline_points, result_img

    def analyze_progress(self):
        # analyze progress
        if len(self.tracking_data["entries"]) < 2:
            return {"message": "Need at least 2 images to analyze progress"}

        # sort entries by date
        sorted_entries = sorted(self.tracking_data["entries"], key=lambda x: x["timestamp"])

        # calculate average hairline distances for each entry
        avg_distances = []
        timestamps = []

        for entry in sorted_entries:
            if entry["hairline_distances"]:
                avg_distance = np.mean(entry["hairline_distances"])
                avg_distances.append(avg_distance)
                timestamps.append(entry["timestamp"])

        # calculate changes
        changes = []
        for i in range(1, len(avg_distances)):
            change = avg_distances[i] - avg_distances[i-1]
            change_percent = (change / avg_distances[i-1]) * 100
            changes.append({
                "from": timestamps[i-1],
                "to": timestamps[i],
                "change_pixels": change,
                "change_percent": change_percent,
                "direction": "Receding" if change < 0 else "Advancing" if change > 0 else "Stable"
            })

        # calculate overall change
        if len(avg_distances) >= 2:
            overall_change = avg_distances[-1] - avg_distances[0]
            overall_percent = (overall_change / avg_distances[0]) * 100
            overall_direction = "Receding" if overall_change < 0 else "Advancing" if overall_change > 0 else "Stable"
        else:
            overall_change = 0
            overall_percent = 0
            overall_direction = "Insufficient data"

        return {
            "entries_analyzed": len(avg_distances),
            "period_start": timestamps[0] if timestamps else None,
            "period_end": timestamps[-1] if timestamps else None,
            "changes": changes,
            "overall_change_pixels": overall_change,
            "overall_change_percent": overall_percent,
            "overall_direction": overall_direction,
            "avg_distances": avg_distances,
            "timestamps": timestamps
        }

    def generate_progress_chart(self):
        # get progress analysis
        analysis = self.analyze_progress()

        if "message" in analysis:
            return None

        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # plot average distances on chart
        dates = [datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S") for ts in analysis["timestamps"]]
        ax.plot(dates, analysis["avg_distances"], 'b-o', linewidth=2)

        # add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Distance (pixels)')
        ax.set_title('Hairline Distance Over Time')

        # add trend line
        if len(dates) > 1:
            z = np.polyfit(range(len(dates)), analysis["avg_distances"], 1)
            p = np.poly1d(z)
            ax.plot(dates, p(range(len(dates))), "r--",
                    label=f"Trend: {p[1]:.2f} pixels/measurement")

        # add change info
        info_text = f"Overall: {analysis['overall_direction']}\n"
        info_text += f"Change: {analysis['overall_change_pixels']:.2f} pixels ({analysis['overall_change_percent']:.2f}%)"
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()

        # save chart to file
        chart_path = os.path.join(self.storage_dir, 'progress_chart.png')
        fig.savefig(chart_path)

        return chart_path

    def create_comparison_image(self, timestamp1, timestamp2):
        # find entries
        entry1 = next((e for e in self.tracking_data["entries"] if e["timestamp"] == timestamp1), None)
        entry2 = next((e for e in self.tracking_data["entries"] if e["timestamp"] == timestamp2), None)
        # if entry not found return None
        if not entry1 or not entry2:
            return None

        # load images
        img1 = cv2.imread(entry1["processed_path"])
        img2 = cv2.imread(entry2["processed_path"])
        # if image not found return None
        if img1 is None or img2 is None:
            return None

        # create comparison image
        comparison = np.hstack((img1, img2))

        # add text labels
        cv2.putText(comparison, timestamp1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(comparison, timestamp2, (img1.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw hairline points from both images
        for point in entry1["hairline_points"]:
            cv2.circle(comparison, point, 3, (0, 255, 0), -1)

        for point in entry2["hairline_points"]:
            # offset x-coordinate by width of first image
            offset_point = (point[0] + img1.shape[1], point[1])
            cv2.circle(comparison, offset_point, 3, (0, 255, 0), -1)

        # save comparison image
        comparison_path = os.path.join(self.storage_dir, f'comparison_{timestamp1}_vs_{timestamp2}.jpg')
        cv2.imwrite(comparison_path, comparison)

        return comparison_path

    def get_all_timestamps(self):
        # returns all image timestamps in chronological order
        timestamps = [entry["timestamp"] for entry in self.tracking_data["entries"]]
        return sorted(timestamps)
