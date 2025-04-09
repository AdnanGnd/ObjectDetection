from helpers import run_inference_on_video

def main():
    # Path to the YOLO model
    model_path = "../params/road_detection/weights/best.pt"

    # Path to the input video file
    input_video_path = "../assets/4608285-uhd_3840_2160_24fps.mp4"
    
    # Path to save the output video file
    output_video_path = "../output/output_video_2.mp4"

    # Run inference on the video
    run_inference_on_video(model_path, input_video_path, output_video_path, conf_threshold=0.5)
    

if __name__ == "__main__":
    main()