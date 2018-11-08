import cv2
cap = cv2.VideoCapture(0)
# cap.set(3, 1600)
# cap.set(4, 600)
while True:
    ret, frame = cap.read()
    if frame is not None:
        cv2.imshow("Frame", frame)
        # print(frame.shape)
        # break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break