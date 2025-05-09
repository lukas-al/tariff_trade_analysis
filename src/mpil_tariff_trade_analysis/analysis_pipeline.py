"""
Analysis Pipeline Script for MPIL Tariff Trade Analysis

Runs the core Marimo analysis scripts sequentially.
"""

import subprocess
from pathlib import Path

from mpil_tariff_trade_analysis.utils.logging_config import get_logger

logger = get_logger(__name__)

ANALYSIS_SCRIPT_DIR = Path(__file__).parent / "analysis"

PIPELINE_SCRIPTS = [
    ANALYSIS_SCRIPT_DIR / "DETREND_PIPELINE.py",
    # Add more analysis scripts here as they are created
]


# --- Main Pipeline Function ---
def run_pipeline():
    """Runs the full analysis pipeline by executing Marimo scripts sequentially."""
    logger.info("--- Starting Analysis Pipeline ---")
    pipeline_successful = True

    for script_path in PIPELINE_SCRIPTS:
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}. Halting pipeline.")
            pipeline_successful = False
            break

        logger.info(f"Running script: {script_path.name}...")
        # Construct the command to run the Marimo script
        # command = [sys.executable, "-m", "marimo", "run", str(script_path)]
        # command = [sys.executable, str(script_path)]
        command = ["uv", "run", str(script_path)]
        logger.debug(f"Executing command: {' '.join(command)}")

        try:
            # Run the script, wait for completion, capture output
            result = subprocess.run(
                command,
                check=True,  # Raise CalledProcessError on non-zero exit code
                # capture_output=True,  # Capture stdout/stderr
                text=True,  # Decode output as text
                encoding="utf-8",  # Specify encoding
            )
            logger.info(f"Successfully finished script: {script_path.name}")
            # logger.debug(f"stdout:\n{result.stdout}")
            logger.debug(f"Res output:\n{result}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running script: {script_path.name}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"stdout:\n{e.stdout}")
            logger.error(f"stderr:\n{e.stderr}")
            pipeline_successful = False
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred while trying to run {script_path.name}: {e}")
            pipeline_successful = False
            break

    # --- Pipeline Completion ---
    if pipeline_successful:
        logger.info("--- Analysis Pipeline finished successfully ---")
    else:
        logger.error("--- Analysis Pipeline finished with errors ---")

    return pipeline_successful  # Return status


# --- Script Execution ---
if __name__ == "__main__":
    run_pipeline()
