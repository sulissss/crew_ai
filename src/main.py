import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered meeting preparation and note-taking using CrewAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prep_parser = subparsers.add_parser(
        "prep",
        help="Prepare for an upcoming meeting by researching participants and industry",
    )
    prep_parser.add_argument(
        "--participants",
        required=True,
        help="Comma-separated emails of meeting participants",
    )
    prep_parser.add_argument(
        "--context",
        required=True,
        help="Context or topic of the meeting",
    )
    prep_parser.add_argument(
        "--objective",
        required=True,
        help="Your objective for the meeting",
    )

    notes_parser = subparsers.add_parser(
        "notes",
        help="Generate structured meeting notes from a transcript",
    )
    notes_parser.add_argument(
        "--transcript",
        required=True,
        help="Path to the meeting transcript file",
    )
    notes_parser.add_argument(
        "--agenda",
        default="General Meeting",
        help="Brief description of the meeting agenda (default: General Meeting)",
    )
    notes_parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable validation agent to verify output quality",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "prep":
        from src.crews.meeting_prep import run_meeting_prep

        print("=" * 60)
        print("  Meeting Preparation Crew")
        print("=" * 60)

        result = run_meeting_prep(args.participants, args.context, args.objective)

        print("\n" + "=" * 60)
        print("  Briefing Document")
        print("=" * 60)
        print(result)

    elif args.command == "notes":
        from src.crews.meeting_notes import run_meeting_notes

        with open(args.transcript, "r") as f:
            transcript = f.read()

        print("=" * 60)
        print("  Meeting Notes Crew")
        print("=" * 60)

        result = run_meeting_notes(transcript, args.agenda, validate=args.validate)

        print("\n" + "=" * 60)
        print("  Meeting Notes")
        print("=" * 60)
        print(result)


if __name__ == "__main__":
    main()
