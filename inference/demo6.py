import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import time

# Streamlit 설정
st.title("Webcam Live Feed with MediaPipe")
run = st.checkbox("Run")
stop = st.checkbox("Stop", value=False)
FRAME_WINDOW = st.image([])
OUTPUT_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# 실제 카메라 FPS 가져오기
actual_fps = camera.get(cv2.CAP_PROP_FPS)
if actual_fps == 0:
    actual_fps = 30  # 웹캠의 FPS를 가져오지 못했을 경우 기본값

# 임시 파일을 통해 동영상 저장
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
output_video_path = temp_file.name

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(
    output_video_path, fourcc, actual_fps, (frame_width, frame_height)
)

# 영상 녹화 상태를 나타내는 플래그
recording = False

if run:
    recording = True

while recording and not stop:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera not accessible.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = frame.copy()
    image.flags.writeable = False

    # MediaPipe 처리
    hands_results = hands.process(image)
    pose_results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손 랜드마크 그리기
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 포즈 랜드마크 그리기
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    FRAME_WINDOW.image(image)

    # 결과를 저장
    out.write(image)
    print(output_video_path)

else:
    st.write("Stopped")
    camera.release()
    out.release()

    if stop:
        cap = cv2.VideoCapture(output_video_path)
        processed_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame.copy()
            image.flags.writeable = False

            # 손 및 포즈 처리
            hands_results = hands.process(image)
            pose_results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 손 랜드마크 그리기
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # 포즈 랜드마크 그리기
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            processed_frames.append(image)

        cap.release()

        # 처리된 결과를 보여주기 (FPS에 맞춰 재생)
        for processed_frame in processed_frames:
            OUTPUT_WINDOW.image(processed_frame)
            time.sleep(1 / actual_fps)

        st.write(f"Output video saved to {output_video_path}")
        st.video(output_video_path)


