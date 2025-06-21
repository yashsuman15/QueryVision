from models.model_task import detection_by_image, detection_by_text
from utils.visualization import plot_detection, display_result

def chose_task():
    while True:
        try:
            user_input = int(input("""Choose a task:\n1. Text-based detection\n2. Image-based detection\nEnter 1 or 2: """))

            match user_input:
                case 1:
                    print("-->DETECTION USING TEXT<--")
                    return detection_by_text
                case 2:
                    print("-->DETECTION USING IMAGE<--")
                    return detection_by_image
                case _:
                    print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


def main():
    
    # choose which task to perform i.e detection_by_text ot detection_by_image
    task = chose_task()
    
    image, boxes, scores, labels = task()
    
    final_result_img = plot_detection(image, boxes, scores, labels)
    
    display_result(final_result_img)
    
    print("---|>MODEL WORKED DONE<|---")

if __name__ == "__main__":
    main()