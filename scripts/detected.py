import cv2
import time
import os
import glob


# ============================================================
# CONFIG
# ============================================================
VIDEO_PATH = "videos/cam1_140012_0121.mp4"
TEMPLATES_GLOB = "templates/*.*"
OUTPUT_VIDEO_PATH = "output/video_anotado.mp4"

TH = 17
COVETE_CANDIDATE_MIN_SCORE = 15

SLEEP_SEC = 0.01

DEBUG_SAVE_MATCHES = True
DEBUG_SAVE_INTERVAL = 2.0

SAVE_INTERVAL_COVETE = 0.0
SAVE_INTERVAL_NAO_COVETE = 3.0
POST_COVETE_BLOCK_SEC = 2.0
DATASET_EVENT_DELAY_SEC = 0.10
NAO_COVETE_MAX_SCORE = 8

SHOW_WINDOW = True
SAVE_OUTPUT_VIDEO = True

# ----------------------------
# ROI
# ----------------------------
ROI_X = 300
ROI_Y = 120
ROI_W = 740
ROI_H = 500

# ----------------------------
# MOTION DETECTION CONFIG
# ----------------------------
MOTION_ENABLED = True
MOTION_BLUR_KERNEL = (21, 21)
MOTION_DIFF_THRESHOLD = 20
MOTION_MIN_AREA = 2500
MOTION_MIN_CHANGED_RATIO = 0.002
MOTION_MAX_CHANGED_RATIO = 0.50
MOTION_REQUIRED_FRAMES = 2
MOTION_RESET_FRAMES = 2
MOTION_HOLD_SEC = 0.8

# ----------------------------
# FRAME SELECTION CONFIG
# ----------------------------
TARGET_CENTER_X_RATIO = 0.38
CENTER_BONUS_MAX = 120.0
MIN_BOX_WIDTH_FOR_CENTER_BONUS = 80
MIN_BOX_HEIGHT_FOR_CENTER_BONUS = 80

SIZE_BONUS_MAX = 120.0
SIZE_BONUS_AREA_REF = 35000.0

SHARPNESS_BONUS_MAX = 80.0
SHARPNESS_REF = 180.0

WEIGHT_CENTER = 1.8
WEIGHT_SIZE = 1.0
WEIGHT_SHARPNESS = 0.7


# ============================================================
# SETUP
# ============================================================
os.makedirs("debug", exist_ok=True)
os.makedirs("dataset/covete", exist_ok=True)
os.makedirs("dataset/nao_covete", exist_ok=True)
os.makedirs("output", exist_ok=True)


# ============================================================
# AUX FUNCTIONS
# ============================================================
def compute_center_bonus(box_x, box_y, box_w, box_h, roi_w):
    if box_w < MIN_BOX_WIDTH_FOR_CENTER_BONUS or box_h < MIN_BOX_HEIGHT_FOR_CENTER_BONUS:
        return 0.0

    target_x = roi_w * TARGET_CENTER_X_RATIO
    box_center_x = box_x + (box_w / 2.0)

    distance = abs(box_center_x - target_x)
    normalized_distance = min(distance / roi_w, 1.0)

    return max(0.0, CENTER_BONUS_MAX * (1.0 - normalized_distance))


def compute_size_bonus(box_w, box_h):
    area = float(box_w * box_h)
    normalized = min(area / SIZE_BONUS_AREA_REF, 1.0)
    return SIZE_BONUS_MAX * normalized


def compute_sharpness_bonus(roi_gray):
    sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    normalized = min(sharpness / SHARPNESS_REF, 1.0)
    return SHARPNESS_BONUS_MAX * normalized


def compute_selection_score(roi_gray, motion_box):
    mx, my, mw, mh = motion_box

    center_bonus = compute_center_bonus(mx, my, mw, mh, ROI_W)
    size_bonus = compute_size_bonus(mw, mh)
    sharpness_bonus = compute_sharpness_bonus(roi_gray)

    selection_score = (
        (center_bonus * WEIGHT_CENTER) +
        (size_bonus * WEIGHT_SIZE) +
        (sharpness_bonus * WEIGHT_SHARPNESS)
    )

    return selection_score, center_bonus, size_bonus, sharpness_bonus


def reset_event_state():
    return {
        "start_time": None,

        "best_visual_score": -1.0,
        "best_visual_frame": None,
        "best_visual_roi_gray": None,
        "best_visual_box": None,
        "best_visual_center_bonus": 0.0,
        "best_visual_size_bonus": 0.0,
        "best_visual_sharpness_bonus": 0.0,

        "best_cov_score": -1.0,
        "best_cov_frame": None,
        "best_cov_roi_gray": None,
        "best_cov_box": None,
        "best_cov_live_score": 0,
        "best_cov_template_name": "-",
    }


def load_templates(templates_glob, orb):
    template_paths = sorted(
        [
            p for p in glob.glob(templates_glob)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
    )

    if not template_paths:
        raise SystemExit("ERRO: nenhum template encontrado na pasta templates")

    templates = []

    for path in template_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Nao foi possivel ler template: {path}")
            continue

        kp, des = orb.detectAndCompute(img, None)
        if des is None or len(kp) == 0:
            print(f"[WARN] Template sem features suficientes: {path}")
            continue

        templates.append({
            "path": path,
            "image": img,
            "kp": kp,
            "des": des
        })
        print(f"[INFO] Template carregado: {path} | kp={len(kp)}")

    if not templates:
        raise SystemExit("ERRO: nenhum template valido com features ORB suficientes")

    return templates


def classify_roi_with_orb(roi_gray, orb, bf, templates):
    kp2, des2 = orb.detectAndCompute(roi_gray, None)

    if des2 is None or kp2 is None or len(kp2) == 0:
        return {
            "score": 0,
            "is_covete": False,
            "template_name": "-",
            "kp_count": 0,
            "good_matches": [],
            "template_ref": None,
            "kp2": None
        }

    best_score = -1
    best_matches = []
    best_template_name = "-"
    best_template_ref = None

    for tpl in templates:
        matches = bf.knnMatch(tpl["des"], des2, k=2)
        good = []

        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        local_score = len(good)

        if local_score > best_score:
            best_score = local_score
            best_matches = good
            best_template_name = os.path.basename(tpl["path"])
            best_template_ref = tpl

    final_score = max(best_score, 0)

    return {
        "score": final_score,
        "is_covete": final_score >= TH,
        "template_name": best_template_name,
        "kp_count": len(kp2),
        "good_matches": best_matches,
        "template_ref": best_template_ref,
        "kp2": kp2
    }


def detect_motion(roi_gray, prev_motion_frame, motion_counter, no_motion_counter, last_motion_time):
    motion_detected = True
    motion_mask = None
    changed_ratio = 0.0
    max_motion_area = 0.0
    main_motion_box = None
    current_motion_frame = cv2.GaussianBlur(roi_gray, MOTION_BLUR_KERNEL, 0)

    if prev_motion_frame is None:
        return {
            "motion_detected": False,
            "motion_mask": None,
            "changed_ratio": 0.0,
            "max_motion_area": 0.0,
            "main_motion_box": None,
            "prev_motion_frame": current_motion_frame,
            "motion_counter": motion_counter,
            "no_motion_counter": no_motion_counter,
            "last_motion_time": last_motion_time,
            "skip_frame": True
        }

    diff = cv2.absdiff(prev_motion_frame, current_motion_frame)

    _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    changed_pixels = cv2.countNonZero(motion_mask)
    changed_ratio = changed_pixels / float(ROI_W * ROI_H)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_motion_area:
            max_motion_area = area
            x, y, w, h = cv2.boundingRect(c)
            main_motion_box = (x, y, w, h)

    raw_motion = (
        max_motion_area >= MOTION_MIN_AREA and
        changed_ratio >= MOTION_MIN_CHANGED_RATIO and
        changed_ratio <= MOTION_MAX_CHANGED_RATIO
    )

    if raw_motion:
        motion_counter += 1
        no_motion_counter = 0
    else:
        no_motion_counter += 1
        if no_motion_counter >= MOTION_RESET_FRAMES:
            motion_counter = 0

    if motion_counter >= MOTION_REQUIRED_FRAMES:
        last_motion_time = time.time()

    motion_detected = (time.time() - last_motion_time) <= MOTION_HOLD_SEC

    return {
        "motion_detected": motion_detected,
        "motion_mask": motion_mask,
        "changed_ratio": changed_ratio,
        "max_motion_area": max_motion_area,
        "main_motion_box": main_motion_box,
        "prev_motion_frame": current_motion_frame,
        "motion_counter": motion_counter,
        "no_motion_counter": no_motion_counter,
        "last_motion_time": last_motion_time,
        "skip_frame": False
    }


def save_debug_matches(now, roi_gray, final_result):
    if final_result["template_ref"] is None or len(final_result["good_matches"]) == 0:
        return False

    kp2 = final_result["kp2"]
    if kp2 is None:
        return False

    top_matches = sorted(final_result["good_matches"], key=lambda m: m.distance)[:30]

    match_img = cv2.drawMatches(
        final_result["template_ref"]["image"],
        final_result["template_ref"]["kp"],
        roi_gray,
        kp2,
        top_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    status = "COVETE" if final_result["is_covete"] else "NAO_COVETE"
    filename = f"debug/matches_{int(now)}_{status}_{final_result['score']}_{final_result['template_name']}.jpg"
    cv2.imwrite(filename, match_img)
    print(f"[DEBUG] Imagem de matches guardada: {filename}")
    return True


def draw_overlay(
    frame,
    motion_detected,
    live_result,
    motion_mask,
    main_motion_box,
    event_state,
    changed_ratio,
    max_motion_area,
    frame_index,
    last_final_score,
    last_final_template_name,
    last_final_source
):
    draw = frame.copy()

    status = "COVETE" if live_result["is_covete"] else "NAO COVETE"
    motion_status = "MOVIMENTO" if motion_detected else "SEM MOVIMENTO"

    selection_info = "-"
    if event_state["best_visual_frame"] is not None:
        selection_info = f"{event_state['best_visual_score']:.1f}"

    cov_candidate_info = "-"
    if event_state["best_cov_frame"] is not None:
        cov_candidate_info = f"{event_state['best_cov_score']:.1f}"

    text = (
        f"frame={frame_index} | {motion_status} | {status} | "
        f"live_matches={live_result['score']} | tpl={live_result['template_name']} | kp={live_result['kp_count']} | "
        f"best_sel={selection_info} | cov_sel={cov_candidate_info} | "
        f"final={last_final_score} | source={last_final_source} | "
        f"ratio={changed_ratio:.4f} | area={int(max_motion_area)}"
    )

    if motion_detected:
        color = (0, 255, 0) if live_result["is_covete"] else (0, 255, 255)
    else:
        color = (120, 120, 120)

    cv2.rectangle(draw, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), color, 2)

    if motion_detected and motion_mask is not None:
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MOTION_MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(
                draw,
                (ROI_X + x, ROI_Y + y),
                (ROI_X + x + w, ROI_Y + y + h),
                (0, 0, 255),
                2
            )

    if motion_detected and main_motion_box is not None:
        mx, my, mw, mh = main_motion_box
        cv2.rectangle(
            draw,
            (ROI_X + mx, ROI_Y + my),
            (ROI_X + mx + mw, ROI_Y + my + mh),
            (255, 0, 255),
            3
        )

    target_x_full = int(ROI_X + (ROI_W * TARGET_CENTER_X_RATIO))
    cv2.line(draw, (target_x_full, ROI_Y), (target_x_full, ROI_Y + ROI_H), (255, 255, 0), 2)

    if event_state["best_visual_box"] is not None:
        bx, by, bw, bh = event_state["best_visual_box"]
        cv2.rectangle(
            draw,
            (ROI_X + bx, ROI_Y + by),
            (ROI_X + bx + bw, ROI_Y + by + bh),
            (0, 255, 0),
            2
        )

    if event_state["best_cov_box"] is not None:
        bx, by, bw, bh = event_state["best_cov_box"]
        cv2.rectangle(
            draw,
            (ROI_X + bx, ROI_Y + by),
            (ROI_X + bx + bw, ROI_Y + by + bh),
            (255, 255, 0),
            2
        )

    cv2.putText(
        draw,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    extra_text = (
        f"visual(center={event_state['best_visual_center_bonus']:.1f},"
        f"size={event_state['best_visual_size_bonus']:.1f},"
        f"sharp={event_state['best_visual_sharpness_bonus']:.1f}) | "
        f"covcand(score={event_state['best_cov_live_score']},tpl={event_state['best_cov_template_name']}) | "
        f"last_tpl={last_final_template_name}"
    )

    cv2.putText(
        draw,
        extra_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return draw, text


# ============================================================
# LOAD VIDEO / TEMPLATES
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"ERRO: não foi possível abrir o vídeo '{VIDEO_PATH}'")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0 or video_fps != video_fps:
    video_fps = 20.0

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video: {VIDEO_PATH}")
print(f"[INFO] FPS: {video_fps:.2f}")
print(f"[INFO] Resolucao: {frame_width}x{frame_height}")

if ROI_X < 0 or ROI_Y < 0 or ROI_X + ROI_W > frame_width or ROI_Y + ROI_H > frame_height:
    raise SystemExit(
        f"ERRO: ROI fora dos limites da imagem. "
        f"Imagem={frame_width}x{frame_height}, ROI=({ROI_X},{ROI_Y},{ROI_W},{ROI_H})"
    )

orb = cv2.ORB_create(nfeatures=1200)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
templates = load_templates(TEMPLATES_GLOB, orb)

writer = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        video_fps,
        (frame_width, frame_height)
    )


# ============================================================
# STATE
# ============================================================
last_debug_save = 0.0
last_save_covete = 0.0
last_save_nao_covete = 0.0
last_covete_detect_time = 0.0
last_print = 0.0

prev_motion_frame = None
motion_counter = 0
no_motion_counter = 0
last_motion_time = 0.0

prev_motion_detected = False
event_state = reset_event_state()

last_final_score = 0
last_final_template_name = "-"
last_final_source = "-"

frame_index = 0

print("\n[INFO] A processar video offline... prima 'q' ou ESC para parar.\n")


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Fim do video.")
        break

    frame_index += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    motion_data = detect_motion(
        roi_gray,
        prev_motion_frame,
        motion_counter,
        no_motion_counter,
        last_motion_time
    )

    prev_motion_frame = motion_data["prev_motion_frame"]
    motion_counter = motion_data["motion_counter"]
    no_motion_counter = motion_data["no_motion_counter"]
    last_motion_time = motion_data["last_motion_time"]

    if motion_data["skip_frame"]:
        continue

    motion_detected = motion_data["motion_detected"]
    motion_mask = motion_data["motion_mask"]
    changed_ratio = motion_data["changed_ratio"]
    max_motion_area = motion_data["max_motion_area"]
    main_motion_box = motion_data["main_motion_box"]

    now = time.time()

    live_result = {
        "score": 0,
        "is_covete": False,
        "template_name": "-",
        "kp_count": 0,
        "good_matches": [],
        "template_ref": None,
        "kp2": None
    }

    if motion_detected:
        live_result = classify_roi_with_orb(roi_gray, orb, bf, templates)

    # ----------------------------
    # INICIO DE EVENTO
    # ----------------------------
    if not prev_motion_detected and motion_detected:
        print(f"[EVENT] INICIO | frame={frame_index}")
        event_state = reset_event_state()
        event_state["start_time"] = now

    # ----------------------------
    # ESCOLHER MELHOR FRAME DO EVENTO
    # ----------------------------
    allow_event_evaluation = (
        motion_detected and
        event_state["start_time"] is not None and
        (now - event_state["start_time"]) >= DATASET_EVENT_DELAY_SEC and
        main_motion_box is not None
    )

    if allow_event_evaluation:
        selection_score, center_bonus, size_bonus, sharpness_bonus = compute_selection_score(
            roi_gray,
            main_motion_box
        )

        if selection_score > event_state["best_visual_score"]:
            event_state["best_visual_score"] = selection_score
            event_state["best_visual_frame"] = frame.copy()
            event_state["best_visual_roi_gray"] = roi_gray.copy()
            event_state["best_visual_box"] = main_motion_box
            event_state["best_visual_center_bonus"] = center_bonus
            event_state["best_visual_size_bonus"] = size_bonus
            event_state["best_visual_sharpness_bonus"] = sharpness_bonus

        if live_result["score"] >= COVETE_CANDIDATE_MIN_SCORE:
            if selection_score > event_state["best_cov_score"]:
                event_state["best_cov_score"] = selection_score
                event_state["best_cov_frame"] = frame.copy()
                event_state["best_cov_roi_gray"] = roi_gray.copy()
                event_state["best_cov_box"] = main_motion_box
                event_state["best_cov_live_score"] = live_result["score"]
                event_state["best_cov_template_name"] = live_result["template_name"]

    # ----------------------------
    # FIM DE EVENTO
    # ----------------------------
    if prev_motion_detected and not motion_detected:
        print(f"[EVENT] FIM | frame={frame_index} | best_sel={event_state['best_visual_score']:.1f}")

        frame_to_classify = event_state["best_visual_frame"]
        roi_to_classify = event_state["best_visual_roi_gray"]
        source_to_classify = "visual"
        selected_selection_score = event_state["best_visual_score"]

        if event_state["best_cov_frame"] is not None and event_state["best_cov_roi_gray"] is not None:
            frame_to_classify = event_state["best_cov_frame"]
            roi_to_classify = event_state["best_cov_roi_gray"]
            source_to_classify = "cov_candidate"
            selected_selection_score = event_state["best_cov_score"]

        if frame_to_classify is not None and roi_to_classify is not None:
            final_result = classify_roi_with_orb(roi_to_classify, orb, bf, templates)

            last_final_score = final_result["score"]
            last_final_template_name = final_result["template_name"]
            last_final_source = source_to_classify

            if final_result["is_covete"]:
                if now - last_save_covete >= SAVE_INTERVAL_COVETE:
                    filename = f"dataset/covete/covete_{int(now * 1000)}.jpg"
                    cv2.imwrite(filename, frame_to_classify)

                    print(
                        f"[DATASET] Guardado em covete: {filename} | "
                        f"final_score={final_result['score']} | "
                        f"selection={selected_selection_score:.1f} | "
                        f"source={source_to_classify} | "
                        f"tpl={final_result['template_name']} | kp={final_result['kp_count']}"
                    )

                    last_save_covete = now
                    last_covete_detect_time = now
            else:
                can_save_nao_covete = (
                    final_result["score"] <= NAO_COVETE_MAX_SCORE and
                    (now - last_covete_detect_time) >= POST_COVETE_BLOCK_SEC
                )

                if can_save_nao_covete and (now - last_save_nao_covete >= SAVE_INTERVAL_NAO_COVETE):
                    filename = f"dataset/nao_covete/nao_covete_{int(now * 1000)}.jpg"
                    cv2.imwrite(filename, frame_to_classify)

                    print(
                        f"[DATASET] Guardado em nao_covete: {filename} | "
                        f"final_score={final_result['score']} | "
                        f"selection={selected_selection_score:.1f} | "
                        f"source={source_to_classify} | kp={final_result['kp_count']}"
                    )

                    last_save_nao_covete = now

            if (
                DEBUG_SAVE_MATCHES and
                (now - last_debug_save >= DEBUG_SAVE_INTERVAL)
            ):
                if save_debug_matches(now, roi_to_classify, final_result):
                    last_debug_save = now

        event_state = reset_event_state()

    prev_motion_detected = motion_detected

    # ----------------------------
    # OVERLAY / PRINT
    # ----------------------------
    draw, text = draw_overlay(
        frame=frame,
        motion_detected=motion_detected,
        live_result=live_result,
        motion_mask=motion_mask,
        main_motion_box=main_motion_box,
        event_state=event_state,
        changed_ratio=changed_ratio,
        max_motion_area=max_motion_area,
        frame_index=frame_index,
        last_final_score=last_final_score,
        last_final_template_name=last_final_template_name,
        last_final_source=last_final_source
    )

    if now - last_print >= 0.2:
        print(text)
        last_print = now

    if SAVE_OUTPUT_VIDEO and writer is not None:
        writer.write(draw)

    if SHOW_WINDOW:
        cv2.imshow("Detecao de Covete - Video", draw)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            print("[INFO] Interrompido pelo utilizador.")
            break

    time.sleep(SLEEP_SEC)


# ============================================================
# CLEANUP
# ============================================================
cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()
print("[INFO] Processamento concluido.")