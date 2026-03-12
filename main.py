import argparse
import sys
import os
import logging
from pathlib import Path

# Dual Logger: Write standard output to BOTH terminal and log file
class LoggerTee:
    """Intercepte les flux (stdout/stderr) pour écrire dans un fichier ET dans la console."""
    def __init__(self, filename: Path, stream):
        self.stream = stream
        self.log_file = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def setup_logger(log_file_path: Path = None):
    # Pre-configuration avant l'éventuel override
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = LoggerTee(log_file_path, sys.stdout)
        sys.stderr = LoggerTee(log_file_path, sys.stderr)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout, 
            force=True
        )

# Make sure we can find custom modules no matter where we are invoked from
FILEPATH = Path(__file__).resolve()
REPO_ROOT = FILEPATH.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.database.cli_ops import run_pseudonymization_cli
from src.extraction.cli_ops import run_extraction_cli
from src.ui.app_qt import main as app_qt_main

def main():
    parser = argparse.ArgumentParser(description="BRIGHT Clinical Database Main Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Pseudo Command
    pseudo_parser = subparsers.add_parser("pseudo", help="Create/Update Pseudonymized DB from PDFs")
    pseudo_parser.add_argument("--gui", action="store_true", help="Launch the QT GUI. Other CLI arguments will be ignored.")
    pseudo_parser.add_argument("--db", type=Path, help="Path to the document database CSV")
    pseudo_parser.add_argument("--pdfs", type=Path, help="Folder containing PDFs or a specific PDF file")
    pseudo_parser.add_argument(
        "--eds_path",
        type=Path,
        default=None,
        help="Path to the HuggingFace cache directory containing the eds-pseudo model"
    )
    pseudo_parser.add_argument("--no_pseudo_only", action="store_true", help="Do not create a _pseudo_only.csv copy")

    # Extract Command
    extract_parser = subparsers.add_parser("extract", help="Extract clinical/bio features to output folder")
    extract_parser.add_argument("--db", required=True, type=Path, help="Path to the pseudonymized document database CSV")
    extract_parser.add_argument("--output", required=True, type=Path, help="Path to output directory (created if missing)")
    extract_parser.add_argument("--use-gliner", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable GLiNER extraction")
    extract_parser.add_argument("--batching-strategy", type=str, default="heterogeneous", choices=["semantic_context", "semantic_only", "heterogeneous"])
    extract_parser.add_argument("--parallel", type=int, default=os.cpu_count()-2, help="Number of workers for parallel processing")
    extract_parser.add_argument("--use-disambiguator", action=argparse.BooleanOptionalAction, default=False, help="Enable textual context disambiguation before GLiNER")

    args, qt_args = parser.parse_known_args()

    if args.command == "pseudo":
        if args.gui:
            # We let the app handle sys.argv naturally by delegating
            sys.argv = [sys.argv[0]] + qt_args
            if args.eds_path:
                sys.argv.extend(["--eds_path", str(args.eds_path.resolve())])
            print("\nStarting application...")
            app_qt_main()
        else:
            if not args.db or not args.pdfs:
                print("Error: --db and --pdfs are required when not using --gui")
                pseudo_parser.print_help()
                sys.exit(1)
            
            setup_logger(args.db.resolve().parent / "cli_pseudo_log.txt")
            
            db_path = args.db.resolve()
            pdfs_path = args.pdfs.resolve()
            eds_path = args.eds_path.resolve() if args.eds_path else None
            
            if pdfs_path.is_dir():
                pdf_files = sorted(pdfs_path.glob("*.pdf"))
            elif pdfs_path.is_file() and pdfs_path.suffix.lower() == ".pdf":
                pdf_files = [pdfs_path]
            else:
                print(f"Error: {pdfs_path} is neither a directory nor a PDF file")
                sys.exit(1)

            run_pseudonymization_cli(
                db_path=db_path,
                pdf_paths=pdf_files,
                eds_path=eds_path,
                make_pseudo_only=not args.no_pseudo_only
            )

    elif args.command == "extract":
        db_path = args.db.resolve()
        output_dir = args.output.resolve()
        
        setup_logger(output_dir / "cli_extraction_log.txt")
        run_extraction_cli(
            db_path=db_path,
            output_dir=output_dir,
            use_gliner=args.use_gliner,
            batching_strategy=args.batching_strategy,
            parallel_workers=args.parallel,
            use_disambiguator=args.use_disambiguator,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
