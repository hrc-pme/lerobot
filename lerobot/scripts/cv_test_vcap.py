import cv2

VIDEO_DEVICE = 28

def main():
    # Set the video capture device to VIDEO_DEVICE
    video_capture = cv2.VideoCapture(VIDEO_DEVICE)

    if not video_capture.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Video Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()