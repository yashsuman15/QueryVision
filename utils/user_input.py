from models.model_loader import Owlv2ModelLoader, OwlvitModelLoader

def choose_model():
    while True:
        try:
            user_input = int(input("""Choose a model:\n1. OWLv2\n2. OWLvit\nEnter 1 or 2: """))

            match user_input:
                case 1:
                    print("---| OWLv2 MODEL CHOOSED |---")
                    return Owlv2ModelLoader()
                case 2:
                    print("---| OWLvit MODEL CHOOSED |---")
                    return OwlvitModelLoader()
                case _:
                    print("Invalid choice. Please select 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        
            
def get_queries(prompt: str) -> list[str]:
    """
    Prompts for a comma-separated list of strings, then:
    - Splits on commas
    - Strips leading/trailing whitespace
    - Discards empty items
    - Repeats until at least one valid item is entered
    
    Returns:
        List of non-empty, stripped strings.
    """
    while True:
        raw = input(prompt)
        if not isinstance(raw, str):
            print("⚠️  Input must be a string. Please try again.")
            continue

        # Split, strip, and filter out empty strings
        items = [item.strip() for item in raw.split(",")]
        items = [item for item in items if item]

        if not items:
            print("⚠️  You must enter at least one item. Please try again.")
        else:
            return items