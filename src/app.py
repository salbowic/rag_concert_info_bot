import sys
from utils.streamlit_ui import run_streamlit_ui
from utils.cli import run_cli

if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        # Run the Streamlit UI by default
        run_streamlit_ui()