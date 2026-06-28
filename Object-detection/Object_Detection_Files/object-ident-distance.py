import cv2
import math

# Threshold to detect object (increased from 0.45 to 0.55 to prevent false positives)
thres = 0.55 

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# ----------------- DISTANCE ESTIMATION SETUP -----------------
FOCAL_LENGTH = 600.0  

KNOWN_WIDTHS = {
    "cup": 8.0,          # Average coffee mug diameter
    "bottle": 7.0,       # Average water bottle width
    "cell phone": 7.5,   # Average smartphone width
    "person": 45.0,      # Average shoulder width
    "laptop": 35.0,      # Average laptop width
    "book": 15.0,        # Average book width
    "chair": 50.0,       # Average chair width
}
# -------------------------------------------------------------

# ----------------- TEMPORAL TRACKER & SMOOTHER ---------------
class CentroidTracker:
    def __init__(self, max_lost_frames=3, min_detect_frames=3, alpha=0.55):
        self.next_id = 0
        # self.objects: id -> {class, box, centroid, lost_frames, consecutive_frames, confidence}
        self.objects = {}       
        self.max_lost_frames = max_lost_frames
        self.min_detect_frames = min_detect_frames
        self.alpha = alpha  # Bbox smoothing weight (higher = faster update, lower = smoother)

    def update(self, detections):
        # detections is a list of tuples: (box, class_name, confidence)
        
        # Increment lost frames count for all tracked objects
        for obj_id in list(self.objects.keys()):
            self.objects[obj_id]["lost_frames"] += 1
            
        if not detections:
            self.clean_up()
            return self.get_tracked_objects()
            
        # Parse inputs
        input_centroids = []
        input_boxes = []
        input_classes = []
        input_confs = []
        for box, class_name, conf in detections:
            x, y, w, h = box
            cx = x + w / 2
            cy = y + h / 2
            input_centroids.append((cx, cy))
            input_boxes.append(box)
            input_classes.append(class_name)
            input_confs.append(conf)
            
        # Register new objects if none exist yet
        if not self.objects:
            for i in range(len(input_centroids)):
                self.register(input_boxes[i], input_classes[i], input_confs[i])
        else:
            object_ids = list(self.objects.keys())
            used_ids = set()
            used_inputs = set()
            
            # Match detections to existing objects using centroid distance and class checks
            for i, (icx, icy) in enumerate(input_centroids):
                best_dist = float('inf')
                best_id = None
                
                for oid in object_ids:
                    if oid in used_ids or self.objects[oid]["class"] != input_classes[i]:
                        continue
                    
                    ocx, ocy = self.objects[oid]["centroid"]
                    dist = math.sqrt((icx - ocx)**2 + (icy - ocy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = oid
                
                # Match threshold: 120 pixels max movement between consecutive frames
                if best_id is not None and best_dist < 120:
                    prev_box = self.objects[best_id]["box"]
                    new_box = input_boxes[i]
                    
                    # Smooth box coordinates using Exponential Moving Average
                    smoothed_box = [
                        int(self.alpha * new_box[j] + (1 - self.alpha) * prev_box[j])
                        for j in range(4)
                    ]
                    
                    self.objects[best_id]["box"] = smoothed_box
                    self.objects[best_id]["centroid"] = (icx, icy)
                    self.objects[best_id]["lost_frames"] = 0
                    self.objects[best_id]["consecutive_frames"] += 1
                    self.objects[best_id]["confidence"] = input_confs[i]
                    
                    used_ids.add(best_id)
                    used_inputs.add(i)
            
            # Register unmatched detections as new objects
            for i in range(len(input_centroids)):
                if i not in used_inputs:
                    self.register(input_boxes[i], input_classes[i], input_confs[i])
                    
        self.clean_up()
        return self.get_tracked_objects()

    def register(self, box, class_name, confidence):
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        self.objects[self.next_id] = {
            "class": class_name,
            "box": box,
            "centroid": (cx, cy),
            "lost_frames": 0,
            "consecutive_frames": 1,
            "confidence": confidence
        }
        self.next_id += 1

    def clean_up(self):
        for oid in list(self.objects.keys()):
            if self.objects[oid]["lost_frames"] > self.max_lost_frames:
                del self.objects[oid]

    def get_tracked_objects(self):
        results = []
        for oid, obj in self.objects.items():
            # Hysteresis: only return objects confirmed over minimum consecutive frames
            # Persistence: keep drawing object temporarily even if temporarily lost
            if obj["consecutive_frames"] >= self.min_detect_frames and obj["lost_frames"] <= self.max_lost_frames:
                results.append((obj["box"], obj["class"], obj["confidence"]))
        return results
# -------------------------------------------------------------

def getObjects(img, thres, nms, draw=True, objects=[], tracker=None):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    
    raw_detections = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if len(objects) == 0 or className in objects:
                raw_detections.append((box, className, confidence))
                
    # Filter detections through temporal tracker if provided
    if tracker is not None:
        tracked_detections = tracker.update(raw_detections)
    else:
        tracked_detections = raw_detections
        
    objectInfo = []
    for box, className, confidence in tracked_detections:
        w_px = box[2]
        distance = None
        if className in KNOWN_WIDTHS:
            distance = (KNOWN_WIDTHS[className] * FOCAL_LENGTH) / w_px
            
        objectInfo.append([box, className, distance])
        
        if draw:
            # Draw bounding box
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            
            # Display Class Name & Confidence
            label = f"{className.upper()} {round(confidence * 100, 1)}%"
            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 0), 2)
            
            # Display Estimated Distance
            if distance is not None:
                dist_label = f"Distance: {round(distance, 1)} cm"
                cv2.putText(img, dist_label, (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)

    return img, objectInfo


if __name__ == "__main__":
    print("Select Video Source:")
    print("1. Standard Webcam (USB / Built-in)")
    print("2. DroidCam (IP Stream)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        ip = input("Enter DroidCam IP (e.g. 192.168.1.50): ").strip()
        port = input("Enter DroidCam Port (default 4747): ").strip()
        if not port:
            port = "4747"
        
        # DroidCam typically streams at http://<ip>:<port>/video
        video_source = f"http://{ip}:{port}/video"
        print(f"Connecting to DroidCam at: {video_source}")
    else:
        video_source = 0
        print("Connecting to default webcam...")
        
    cap = cv2.VideoCapture(video_source)
    
    # Set resolution only if using a local hardware camera index
    if isinstance(video_source, int):
        cap.set(3, 640)
        cap.set(4, 480)
        
    # Initialize the temporal tracker
    tracker = CentroidTracker(max_lost_frames=3, min_detect_frames=3, alpha=0.55)
    
    print("Press 'q' in the camera window to quit.")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera.")
            break
            
        # Use our tracker to process detections
        result, objectInfo = getObjects(img, thres, 0.2, tracker=tracker)
        
        cv2.imshow("Object Detection & Distance Estimation", img)
        
        # Stop script if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
