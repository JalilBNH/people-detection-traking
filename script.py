from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import supervision as sv


def draw_squares(frame, bbox, track_id):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color=(255, 0, 0),
        thickness=2
    )
    
    cv2.putText(
        frame,
        f'{track_id}',
        (int(x1 - 15), int(y1 + 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0)
    )
    return frame


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        30,
        (frames[0].shape[1], frames[0].shape[0])
    )
    
    for frame in frames:
        out.write(frame)
    out.release



model = YOLO('yolo11m.pt') # N'oubliez pas de télécharger le modèle pour tester le script
tracker = sv.ByteTrack()

frames = read_video('./test_videos/video_2.mp4')
detections = model.predict(frames)

tracks= {
    'person' : [],
}

for frame_num, detection in enumerate(detections):
    class_names = detection.names
    class_names_inv = {v:k for k,v in class_names.items()}
    
    detection_sv = sv.Detections.from_ultralytics(detection)
    tracks['person'].append({})
    
    detection_tracks = tracker.update_with_detections(detection_sv)
    
    for frame_detection in detection_tracks:
        bbox = frame_detection[0].tolist()
        class_id = frame_detection[3]
        track_id = frame_detection[4]
        
        if class_id == class_names_inv['person']:
            tracks['person'][frame_num][int(track_id)] = {'bbox': bbox}


color_dict = {0: (255, 0, 0)}
output_video_frames = []
for frame_num, frame in enumerate(frames):
    frame = frame.copy()
    
    person_dict = tracks['person'][frame_num]
    
    for track_id, person in person_dict.items():
        bbox = person['bbox']
        frame = draw_squares(frame, person['bbox'], track_id)
        
    output_video_frames.append(frame)
    
    
save_video(output_video_frames, './result_videos/video_out2.mp4')