
import cv2
from pytesseract import pytesseract
from pytesseract import Output
import os,sys
import numpy as np
from numpy_ringbuffer import RingBuffer


# params for question box
QN_H_START_PERC = 840 / 1080
QN_H_END_PERC = 970 / 1080
QN_W_START_PERC = 190 / 1920
QN_W_END_PERC = 1410 / 1920
QN_BOX_EFF_AREA_FACT_LO = 0.7881
QN_BOX_EFF_AREA_FACT_HI = 0.8196
N_QN_CONT_EDGES = 5

# params for answer characters
MAX_ANS_CHARS = 10
ANS_MAIN_CONT_X_PERC = np.array([292, 340, 390, 390, 340, 292]) / 1920
ANS_MAIN_CONT_Y_PERC = np.array([706, 682, 706, 762, 786, 762]) / 1080
ANS_CHAR_SPACING_PERC = 124 / 1920
ANS_CHAR_HEIGHT_THRESH_HI = 66 / 1080
ANS_CHAR_HEIGHT_THRESH_LO = 48 / 1080

# state for reveal/unreveal phases
WAIT_QN = 0
QN_ARRIVED = 1
REVEAL_FRAME_OFFSET = 36

# capacity of ring buffer
RBUF_CAPA = 25


def create_charbox_contours(img_h, img_w):
    contours = []
    for i in range(MAX_ANS_CHARS):
        x_offset = i * ANS_CHAR_SPACING_PERC
        cnt = [[(x + x_offset) * img_w, y * img_h] for x, y in zip(ANS_MAIN_CONT_X_PERC, ANS_MAIN_CONT_Y_PERC)]
        contours.append(cnt)
    return np.array(contours, dtype=np.int32)



def find_pos_qn(img):
    img_h, img_w = img.shape[0], img.shape[1]
    qn_box = img[int(img_h * QN_H_START_PERC):int(img_h * QN_H_END_PERC),
                           int(img_w * QN_W_START_PERC):int(img_w * QN_W_END_PERC)]
    return qn_box


def detect_text(img, ocr_mode=3, string_mode=True):
    config = '-l eng+tur --oem 0 --psm ' + str(ocr_mode)
    if string_mode:
        return pytesseract.image_to_string(img, config=config)
    boxes_dict = pytesseract.image_to_boxes(img, config=config, output_type=Output.DICT)
    chars = []
    if boxes_dict:
        for i in range(len(boxes_dict["char"])):
            c_width = boxes_dict["right"][i] - boxes_dict["left"][i]
            c_height = boxes_dict["top"][i] - boxes_dict["bottom"][i]
            char = boxes_dict["char"][i]
            chars.append({'char': char, 'width':c_width, 'height':c_height})
    return chars

def detect_stable_box(img, thres_lo, thres_hi, n_edges):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        box_area = cv2.contourArea(contour)
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if (thres_hi > box_area > thres_lo) and (approx.shape[0] == n_edges):
            return True, img
    return False, None

def convert_img_to_bin_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return bin_img

def read_qn(frame):
    bin_f = convert_img_to_bin_img(frame)
    qn_box = find_pos_qn(bin_f)
    qn_text= detect_text(qn_box, ocr_mode=3, string_mode=True)
    return qn_text.replace('\n',' ')

def read_ans(frame):
    f_copy = frame.copy()
    f_h, f_l = f_copy.shape[0], f_copy.shape[1]
    contours = create_charbox_contours(f_h, f_l)
    fill_color = [255, 255, 255]  # any BGR color value to fill with
    mask_value = 255  # 1 channel white (can be any non-zero uint8 value)

    # image masking
    stencil = np.zeros(f_copy.shape[:-1]).astype(np.uint8)
    cv2.fillPoly(stencil, contours, mask_value)
    sel = stencil != mask_value  # select everything that is not mask_value
    f_copy[sel] = fill_color  # and fill it with fill_color
    bin_f = ~convert_img_to_bin_img(f_copy)
    ans_chars = detect_text(bin_f, ocr_mode=7, string_mode=False)
    ans = ''
    valid_chars = []
    for char in ans_chars:
        if ANS_CHAR_HEIGHT_THRESH_HI * f_h > char['height'] > ANS_CHAR_HEIGHT_THRESH_LO * f_h:
            valid_chars.append(char['char'])
    return ans.join(valid_chars)


if __name__ == '__main__':
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    video_file = sys.argv[1]

    cap = cv2.VideoCapture(video_file)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_start_pos = int(sys.argv[2]) * fps if len(sys.argv) > 2 and sys.argv[2] else 0
    vid_end_pos = int(sys.argv[3]) * fps if len(sys.argv) > 3 and sys.argv[3] else n_total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, vid_start_pos)
    curr_frame_idx = vid_start_pos
    state = WAIT_QN
    is_box_stable_arr = RingBuffer(capacity=RBUF_CAPA, dtype=bool)
    ans_qns_arr = []
    ans_qns_frame_arr = []

    while cap.isOpened():
        print(f"Processing frame {curr_frame_idx} of total {int(n_total_frames)}")
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Stream has reached the end...")
            break
        bin_img = convert_img_to_bin_img(frame)
        qn_box = find_pos_qn(bin_img)
        qn_eff_area = (QN_H_END_PERC - QN_H_START_PERC) * bin_img.shape[0] * (QN_W_END_PERC - QN_W_START_PERC) * \
                      bin_img.shape[1]
        box_stable, stable_img = detect_stable_box(qn_box, qn_eff_area * QN_BOX_EFF_AREA_FACT_LO,
                                                   qn_eff_area * QN_BOX_EFF_AREA_FACT_HI, N_QN_CONT_EDGES)
        is_box_stable_arr.append(box_stable)
        box_appeared = np.all(is_box_stable_arr)
        box_disappeared = not np.any(is_box_stable_arr)

        if state == WAIT_QN:
            if box_appeared:
                state = QN_ARRIVED
        elif state == QN_ARRIVED:
            if box_disappeared:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx - REVEAL_FRAME_OFFSET)
                _, f = cap.read()
                qn_text = read_qn(f)
                ans_text = read_ans(f)
                qn_idx = len(ans_qns_arr) + 1
                result = f"Question {qn_idx}: {qn_text}\nAnswer {qn_idx}: {ans_text}\n\n"
                print(result)
                ans_qns_arr.append(result)
                ans_qns_frame_arr.append(f)
                cv2.imshow('frame', f)
                print("Press 'q' to abort...")
                if cv2.waitKey(2000) == ord('q'):
                    print("User aborted!")
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
                state = WAIT_QN

        curr_frame_idx += 1
        if curr_frame_idx > vid_end_pos:
            break
    print("Saving results in output folder...")
    # delete all files in the output folder
    output_path = "./output"
    for file in os.listdir(output_path):
        os.remove(f"{output_path}/{file}")

    # save text outputs and screenshots
    with open(f"{output_path}/questions_answers.txt", mode="w", encoding='utf-8') as file:
        file.writelines(ans_qns_arr)
    for idx, f in enumerate(ans_qns_frame_arr):
        # downsize the image and save
        f_small = cv2.resize(f, (960, 540), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f"{output_path}/qn_ans_{str(idx + 1)}.jpg", f_small)

    cap.release()
    cv2.destroyAllWindows()











