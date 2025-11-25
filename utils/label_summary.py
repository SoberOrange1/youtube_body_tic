import os
import json

def process_annotations(input_folder):
    """
    Process all annotation JSON files in the given folder.
    For each video, sum up tic and non-tic frame counts and print results.
    
    :param input_folder: Path to the folder containing annotation JSON files
    """
    # Iterate over all files in the folder
    for fname in os.listdir(input_folder):
        if fname.endswith(".json") and fname.startswith("annotations_"):
            file_path = os.path.join(input_folder, fname)
            
            # Load JSON content
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Each file has one video entry, key is the video_id
            for video_id, video_data in data.items():
                tic_count = 0
                non_tic_count = 0
                
                # Sum up frame counts by label
                for ann in video_data.get("frame_annotations", []):
                    if ann["label"] == "tic":
                        tic_count += ann["frame_count"]
                    elif ann["label"] == "non-tic":
                        non_tic_count += ann["frame_count"]
                
                # Print results to console
                print(f"Video {video_id}: Tic frames = {tic_count}, Non-tic frames = {non_tic_count}")


# Example usage
if __name__ == "__main__":
    input_folder = r"A:\youtube_body\data_folder\annotations"
    process_annotations(input_folder)
