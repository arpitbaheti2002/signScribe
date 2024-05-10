def image_collection(action):
    import argparse
    import cv2
    import os
    import time
    import uuid

    from yolo import YOLO

    import tkinter as tk
    from tkinter import messagebox

    def ask_to_save_image():
        root = tk.Tk()
        root.withdraw()

        answer = messagebox.askyesno("Save Image", "Do you want to save this image?")

        root.destroy()

        return answer

    no_of_images = 0
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", choices=["normal"],
                    help='Network Type')
    ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
    ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
    args = ap.parse_args()

    if args.network == "normal":
        print("loading yolo...")
        yolo = YOLO("./models/cross-hands.cfg", "./models/cross-hands.weights", ["hand"])

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    print("starting webcam...")
    imgnum = 0
    while no_of_images < 5:
        cv2.namedWindow("Hand Detection")
        vc = cv2.VideoCapture(0)

        rval, frame = vc.read()

        for countdown in range(5, 0, -1):
            countdown_frame = frame.copy()

            cv2.putText(countdown_frame, f'Collecting frames for {action} - Image Number {no_of_images}', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(countdown_frame, f'Next frame in {countdown} seconds', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Hand Detection", countdown_frame)
            cv2.waitKey(1000)
        rval, frame = vc.read()

        width, height, inference_time, results = yolo.inference(frame)
        # sort by confidence
        results.sort(key=lambda x: x[2])

        # how many hands should be shown
        hand_count = len(results)
        if args.hands != -1:
            hand_count = int(args.hands)

        # display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        cv2.imshow("Hand Detection", frame)
        cv2.waitKey(1000)

        if ask_to_save_image():
            filename = f"{action}_{no_of_images}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
        
        no_of_images += 1

    cv2.destroyWindow("Hand Detection")
    vc.release()

    return "Images saved successfully. The model will be trained and implemented after few hours."
