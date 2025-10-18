import cv2
import os

# Correct the video path
video_path = r"/Users/maxs/Documents/University/Master's degree/Scientific work/Bulles/Vidéo des bulles/0_Concentration.mp4"
output_dir = "./extracted_frames"

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The file {video_path} does not exist.")
    exit()

cap = cv2.VideoCapture(video_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break 

    cv2.imwrite(f"{output_dir}/frame{index}.jpg", frame)
    index += 1

cap.release()

print(f"Сохранено {index} кадров в папку {output_dir}.")

def main():
    pass

if __name__ == "__main__":
    main()