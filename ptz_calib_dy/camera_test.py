'''
    RTSP로 프레임을 읽어 그대로 보여주는 뷰어

    영상 속 OSD 시계 = 카메라가 프레임에 구워 넣은 장치 시각
    오버레이 recv = 이 PC에서 cap.read()로 프레임을 받은 시각
'''
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def load_rtsp_url() -> str:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)

    url = os.getenv("URL", "").strip().strip("'\"")
    if url:
        return url

    user = os.getenv("USER", "").strip().strip("'\"")
    password = os.getenv("PASSWORD", "").strip().strip("'\"")
    port = os.getenv("PORT", "554").strip().strip("'\"")
    channel = os.getenv("CHANNEL", "stream0").strip().strip("'\"")

    if not user or not password:
        raise ValueError(".env에 URL 또는 USER/PASSWORD가 필요합니다.")

    host = os.getenv("HOST", "10.0.110.241").strip().strip("'\"")
    return f"rtsp://{user}:{password}@{host}:{port}/{channel}"


def overlay_recv_time(frame, recv_at: datetime):
    """로컬(서버) 수신 시각을 프레임에 그린다. OSD와 구분되도록 좌상단에 표시."""
    text = f"recv (local): {recv_at.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, 10 + th
    pad = 6
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    rtsp_url = load_rtsp_url()
    print(f"RTSP 연결 시도: {rtsp_url.split('@')[-1]}")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("RTSP 스트림을 열 수 없습니다. URL·네트워크·카메라 상태를 확인하세요.")
        sys.exit(1)

    window_name = "RTSP Stream (cam_02)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(
        "영상 OSD = 카메라 장치 시각, "
        "노란 recv = 이 PC 수신 시각. "
        "'s' 저장, 'q'/ESC 종료."
    )

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            recv_at = datetime.now()
            if not ret or frame is None:
                print("프레임을 읽지 못했습니다. 연결을 종료합니다.")
                break

            overlay_recv_time(frame, recv_at)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                path = f"frame_{count}.jpg"
                cv2.imwrite(path, frame)
                print(f"저장됨: {path} (recv={recv_at.isoformat(timespec='milliseconds')})")
                count += 1
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
