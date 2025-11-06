import cv2 as cv
import numpy as np
from collections import deque

# CONFIG / TUNABLE PARAMETERS
WIDTH = 960           # resize width for speed (change to 640 for faster)
HEIGHT = 540          # resize height
GAUSSIAN_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 150
SLOPE_THRESHOLD = 0.5     # abs(slope) must be > threshold to be considered a lane line
SMOOTHING_FRAMES = 6      # moving average of last N frames for lanes

# deque to hold previous averaged lines for smoothing across frames
left_lines_deque = deque(maxlen=SMOOTHING_FRAMES)
right_lines_deque = deque(maxlen=SMOOTHING_FRAMES)


def resize(img, width=WIDTH, height=HEIGHT):
    return cv.resize(img, (width, height))


def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size=GAUSSIAN_KERNEL):
    return cv.GaussianBlur(img, kernel_size, 0)


def canny(img, low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH):
    return cv.Canny(img, low_threshold, high_threshold)


def region_of_interest(img):
    """Return image masked to polygonal region where lanes are expected"""
    mask = np.zeros_like(img)
    h, w = img.shape[:2]
    # polygon coordinates as fractions of image size (tweak for camera)
    polygon = np.array([[
        (int(0.10 * w), h),
        (int(0.90 * w), h),
        (int(0.60 * w), int(0.60 * h)),
        (int(0.40 * w), int(0.60 * h))
    ]], dtype=np.int32)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(img, mask)


def detect_lines(masked_edges):
    """Hough transform lines (returns Nx1x4 array or None)"""
    lines = cv.HoughLinesP(masked_edges,
                           rho=HOUGH_RHO,
                           theta=HOUGH_THETA,
                           threshold=HOUGH_THRESHOLD,
                           minLineLength=HOUGH_MIN_LINE_LENGTH,
                           maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines


def slope_intercept_from_line(line):
    """Given a line [[x1,y1,x2,y2]] return (slope, intercept)"""
    x1, y1, x2, y2 = line.reshape(4)
    if x2 == x1:
        return None  # vertical line (skip)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def separate_lines(lines, img_shape):
    """Separate Hough lines into left and right lane candidates based on slope and x position."""
    left_candidates = []
    right_candidates = []
    h, w = img_shape[:2]
    if lines is None:
        return left_candidates, right_candidates

    for l in lines:
        si = slope_intercept_from_line(l)
        if si is None:
            continue
        slope, intercept = si
        if abs(slope) < SLOPE_THRESHOLD:
            continue  # ignore near-horizontal lines
        # Compute x at mid-height to help filter ambiguous near-center lines
        y_mid = h * 0.75
        x_mid = (y_mid - intercept) / slope if slope != 0 else None

        # Left lines have negative slope (image coords) and x_mid < w/2
        if slope < 0 and x_mid is not None and x_mid < w * 0.55:
            left_candidates.append((slope, intercept))
        # Right lines have positive slope and x_mid > w/2
        elif slope > 0 and x_mid is not None and x_mid > w * 0.45:
            right_candidates.append((slope, intercept))
    return left_candidates, right_candidates


def average_slope_intercept(candidates):
    """Average a list of (slope, intercept) tuples. Returns (slope, intercept) or None."""
    if len(candidates) == 0:
        return None
    slope_avg = np.mean([s for s, _ in candidates])
    intercept_avg = np.mean([b for _, b in candidates])
    return slope_avg, intercept_avg


def make_line_points(y1, y2, slope_intercept):
    """From slope and intercept, compute integer pixel points x1,y1,x2,y2."""
    slope, intercept = slope_intercept
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1), x2, int(y2))


def smooth_line(new_line, deque_lines):
    """Keep smoothing history; deque holds last N (slope, intercept). Return averaged (slope, intercept)."""
    if new_line is None:
        # still return average of previous if exists
        if len(deque_lines) == 0:
            return None
        return np.mean(np.array(deque_lines), axis=0)
    deque_lines.append(new_line)
    return np.mean(np.array(deque_lines), axis=0)


def draw_lines(img, lines, color=(0, 255, 0), thickness=8):
    if lines is None:
        return
    for x1, y1, x2, y2 in lines:
        cv.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    return cv.addWeighted(initial_img, α, img, β, γ)


def process_frame(frame, visualize=False):
    """Process a single frame (image). Returns frame with lane overlay."""
    # 1. Resize if desired
    img = resize(frame)

    # 2. Preprocess
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    edges = canny(blur)
    masked = region_of_interest(edges)

    # 3. Hough lines
    lines = detect_lines(masked)

    # 4. Separate left/right candidates
    left_cand, right_cand = separate_lines(lines, img.shape)

    # 5. Average each side
    left_avg = average_slope_intercept(left_cand)
    right_avg = average_slope_intercept(right_cand)

    # 6. Smooth across frames
    left_smooth = smooth_line(left_avg, left_lines_deque)
    right_smooth = smooth_line(right_avg, right_lines_deque)

    # 7. Build final line coordinates (extrapolate to bottom and to a horizon)
    h = img.shape[0]
    y1 = h                  # bottom of image
    y2 = int(h * 0.6)       # slightly above center (top of ROI)

    line_image = np.zeros_like(img)

    final_lines = []
    if left_smooth is not None:
        pts = make_line_points(y1, y2, left_smooth)
        if pts is not None:
            final_lines.append(pts)
    if right_smooth is not None:
        pts = make_line_points(y1, y2, right_smooth)
        if pts is not None:
            final_lines.append(pts)

    # draw
    draw_lines(line_image, final_lines, thickness=10)

    # combine
    result = weighted_img(line_image, img, α=0.8, β=1.0, γ=0.0)

    if visualize:
        # return multiple debug images if desired (edges, masked, result)
        return result, edges, masked, line_image

    return result


# --------------------
# MAIN: image or video demo
# --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to input image file (process single image).")
    parser.add_argument("--video", help="Path to video file (or '0' for webcam).")
    parser.add_argument("--show-debug", action="store_true", help="Show intermediate debug windows.")
    args = parser.parse_args()

    if args.image:
        frame = cv.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image {args.image}")
        result, edges, masked, line_img = process_frame(frame, visualize=True)
        cv.imshow("Result", result)
        cv.imshow("Edges", edges)
        cv.imshow("Masked", masked)
        cv.imshow("Line Image", line_img)
        print("Press any key to exit.")
        cv.waitKey(0)
        cv.destroyAllWindows()

    elif args.video:
        vid_src = args.video
        if vid_src == "0":
            vid_src = 0
        cap = cv.VideoCapture(vid_src)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source {args.video}")
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        print(f"Opened video. FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame(frame, visualize=False)

            cv.imshow("Lane Detection", result)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
    else:
        print("Run with --image path/to/road.jpg or --video path/to/video.mp4 (use '0' for webcam).")
