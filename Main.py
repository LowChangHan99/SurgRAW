import argparse
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from tqdm import tqdm  
from Orchestrators import final_orchestrator


def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the agentic orchestration system on an XLSX file."
    )
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help="Path to the input XLSX file with columns: image_path, COT_Process, question_mcq, ground_truth.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save the log text files.",
    )
    args = parser.parse_args()

    # Ensure the log directory exists
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print(f"[INFO] Created log directory: {args.log_dir}")

    # Load the XLSX file
    try:
        df = pd.read_excel(args.xlsx_file)
        print(f"[INFO] Loaded XLSX file: {args.xlsx_file}")
    except Exception as e:
        print(f"[ERROR] Failed to load XLSX file: {e}")
        sys.exit(1)

    # Iterate over each row in the DataFrame with a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        image_path = row["image_path"]
        cot_process = row["COT_Process"]
        question = row["question_mcq"]
        # ground_truth is available if you want to use it later:
        ground_truth = row.get("ground_truth", None)

        # Derive image_name from image_path (remove directory and extension)
        base_name = os.path.basename(image_path)
        image_name, _ = os.path.splitext(base_name)
        # Sanitize the COT_Process string (e.g., replace spaces with underscores)
        COT_FileNamingConvention = str(cot_process).replace(" ", "_")
        # Create the log file name as specified
        log_file_name = f"{image_name}_{COT_FileNamingConvention}_SurgCOT.txt"
        log_file_path = os.path.join(args.log_dir, log_file_name)

        print(f"\n[INFO] Processing row {index+1}:")
        print(f"       Image: {image_path}")
        print(f"       COT_Process: {cot_process}")
        print(f"       Question: {question}")
        print(f"       Log file will be saved to: {log_file_path}")

        # Capture all print output for this orchestration run
        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            try:
                # Run the final orchestrator (passing question and image_path)
                # This call will print various messages as defined in your orchestrator
                final_answer = final_orchestrator(question, image_path)
                print("\nFinal Answer:")
                print(final_answer)
            except Exception as e:
                print(f"[ERROR] Exception occurred during orchestration: {e}")

        # Write the captured output to the log file
        output = log_buffer.getvalue()
        with open(log_file_path, "w") as log_file:
            log_file.write(output)

        print(f"[INFO] Finished processing. Log saved to: {log_file_path}")


if __name__ == "__main__":
    main()
