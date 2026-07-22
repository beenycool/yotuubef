"""
Hybrid Documentary YouTube Shorts Generator
Main entry point for the hybrid documentary workflow with AI-powered deep research.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import asyncio
import logging
import argparse
import json
import sys
from datetime import datetime

from src.enhanced_orchestrator import EnhancedVideoOrchestrator
from src.hybrid_documentary_state_machine import PipelinePhase
from src.config.settings import get_config, setup_logging
from src.utils.cleanup import clear_temp_files, clear_results, clear_logs


class HybridYouTubeGenerator:
    """
    Hybrid documentary YouTube Shorts generator with AI-powered deep research.
    """

    def __init__(self):
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.orchestrator = EnhancedVideoOrchestrator()
        self.logger.info("Hybrid YouTube Generator initialized")

    async def run_hybrid_workflow(
        self,
        project_name: str,
        reddit_url: str = None,
        resume: bool = False,
        phase_override: str = None,
        gemini_report_path: str = None,
        no_upload: bool = False,
        no_auto_research: bool = False,
    ) -> dict:
        """Run hybrid documentary workflow with pause/resume state."""
        try:
            print(
                f"Starting hybrid documentary workflow: {project_name}...", flush=True
            )
            self.logger.info(
                "Starting hybrid workflow project=%s resume=%s phase_override=%s",
                project_name,
                resume,
                phase_override,
            )
            result = await self.orchestrator.process_hybrid_documentary_studio(
                project_name=project_name,
                reddit_url=reddit_url,
                resume=resume,
                phase_override=phase_override,
                gemini_report_path=gemini_report_path,
                no_upload=no_upload,
                no_auto_research=no_auto_research,
            )

            phase = result.get("current_phase", "unknown")
            workspace = result.get("workspace_path", "")
            if result.get("success"):
                if result.get("paused"):
                    print(f"Hybrid workflow paused at phase: {phase}")
                else:
                    print(f"Hybrid workflow complete at phase: {phase}")
                if workspace:
                    print(f"Workspace: {workspace}")
                if result.get("final_script_path"):
                    print(f"Final script: {result['final_script_path']}")
                if result.get("final_video_path"):
                    print(f"Final video: {result['final_video_path']}")
            else:
                self.logger.error("Hybrid workflow failed: %s", result.get("error"))

            return result
        except Exception as exc:
            self.logger.exception("Hybrid workflow failed")
            return {"success": False, "error": str(exc)}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def setup_argparse():
    """Setup and configure command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Documentary YouTube Shorts Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a new hybrid documentary project
  python main.py my_project --reddit-url "https://reddit.com/..."
  
  # Resume a paused project
  python main.py my_project --resume
  
  # Skip auto research and provide a Gemini report manually
  python main.py my_project --no-auto-research --gemini-report path/to/report.txt
  
  # Resume to a specific phase
  python main.py my_project --resume --phase VIDEO_RENDER
  
  # Cleanup temporary files
  python main.py --cleanup
        """,
    )

    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Project name for the hybrid documentary workflow",
    )

    parser.add_argument(
        "--reddit-url",
        help="Optional Reddit URL to seed project context",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved hybrid phase state",
    )
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in PipelinePhase],
        help="Override current hybrid phase before processing",
    )
    parser.add_argument(
        "--gemini-report",
        help="Path to Gemini report text/markdown file",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Disable upload stage metadata for hybrid workflow",
    )
    parser.add_argument(
        "--no-auto-research",
        action="store_true",
        help="Skip auto research; pause at WAIT_FOR_GEMINI_REPORT for manual report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print what would run without executing",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files, results, and logs",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Keep last N project directories when cleaning",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Also clear log files (use with --cleanup)",
    )

    return parser


async def main() -> int:
    """Main entry point. Returns a process exit code."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Handle cleanup
    if args.cleanup:
        print("Cleaning up...", flush=True)
        clear_temp_files()
        clear_results()
        if args.logs:
            clear_logs()
        if args.keep:
            print(f"Keeping last {args.keep} project directories.", flush=True)
        print("Cleanup complete.", flush=True)
        return 0

    # Require project name
    if not args.project:
        parser.print_help()
        print("\nError: No project name specified.")
        print("Example: python main.py my_documentary --reddit-url <url>")
        return 1

    generator = HybridYouTubeGenerator()
    exit_code = 0

    try:
        result = await generator.run_hybrid_workflow(
            project_name=args.project,
            reddit_url=args.reddit_url,
            resume=args.resume,
            phase_override=args.phase,
            gemini_report_path=args.gemini_report,
            no_upload=args.no_upload,
            no_auto_research=args.no_auto_research,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = Path(f"data/results/results_hybrid_{timestamp}.json")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")
        if not result.get("success"):
            print(f"ERROR: Hybrid workflow failed: {result.get('error')}")
            exit_code = 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
        exit_code = 130
    except Exception as e:
        print(f"ERROR: Operation failed: {e}", flush=True)
        logging.getLogger(__name__).exception("Main operation failed")
        exit_code = 1

    return exit_code


def run_main():
    """Safe main entry point with proper async handling"""
    try:
        try:
            loop = asyncio.get_running_loop()
            print("Already in async context. Use 'await main()' instead.", flush=True)
            return 1
        except RuntimeError:
            pass
        code = asyncio.run(main())
        return 0 if code is None else code
    except KeyboardInterrupt:
        print("\nStartup interrupted by user", flush=True)
        return 130
    except Exception as e:
        print(f"Critical startup error: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(run_main())
